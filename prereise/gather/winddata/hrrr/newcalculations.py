import os
import concurrent.futures
import pygrib
import numpy as np
import pandas as pd
import datetime
from powersimdata.utility.distance import ll2uv
from scipy.spatial import KDTree
from tqdm import tqdm
from prereise.gather.winddata import const
from prereise.gather.winddata.hrrr.helpers import formatted_filename
from prereise.gather.winddata.impute import linear
from prereise.gather.winddata.power_curves import (
    get_turbine_power_curves,
    shift_turbine_curve,
)


def log_error(e, filename, hours_forecasted="0", message="in something"):
    """Log error message."""
    print(
        f"In {filename} ERROR: {e} occurred {message} for hours forecast {hours_forecasted}"
    )


def get_wind_data_lat_long(dt, directory, hours_forecasted="0"):
    """Return latitude and longitudes of wind grid sectors."""
    try:
        import pygrib
    except ImportError:
        print("pygrib is missing but required for this function")
        raise

    file_path = os.path.join(
        directory, formatted_filename(dt, hours_forecasted=hours_forecasted)
    )
    gribs = pygrib.open(file_path)
    grib = next(gribs)

    return grib.latlons()


def find_closest_wind_grids(wind_farms, wind_data_lat_long):
    """Calculate the closest wind grid to each wind farm."""
    grid_lats, grid_lons = (
        wind_data_lat_long[0].flatten(),
        wind_data_lat_long[1].flatten(),
    )
    grid_lat_lon_unit_vectors = [
        ll2uv(lon, lat) for lon, lat in zip(grid_lons, grid_lats)
    ]

    tree = KDTree(grid_lat_lon_unit_vectors)
    wind_farm_unit_vectors = [
        ll2uv(lon, lat)
        for lon, lat in zip(wind_farms.lon.values, wind_farms.lat.values)
    ]

    _, indices = tree.query(wind_farm_unit_vectors)
    return indices


def get_shifted_curve(lookup_name, hub_height, reference_curves):
    """Get the power curve for the given turbine type, shifted to hub height.
    """
    return shift_turbine_curve(
        reference_curves[lookup_name],
        hub_height,
        const.max_wind_speed,
        const.new_curve_res,
    )


def calculate_pout_individual(wind_speed_data, wind_farms, start_dt, end_dt, directory):
    """Calculate power output for wind farms based on hrrr data."""
    turbine_power_curves = get_turbine_power_curves()

    def get_starting_curve_name(series):
        full_name = " ".join([series[const.mfg_col], series[const.model_col]])
        return full_name if full_name in turbine_power_curves.columns else "IEC class 2"

    lookup_names = wind_farms.apply(get_starting_curve_name, axis=1)
    lookup_values = pd.concat(
        [lookup_names, wind_farms[const.hub_height_col]], axis=1)

    shifted_power_curves = lookup_values.apply(
        lambda x: get_shifted_curve(*x), axis=1)

    wind_power_data = [
        [
            np.interp(
                wind_speed_data.loc[dt, w],
                shifted_power_curves.loc[w].index.values,
                shifted_power_curves.loc[w].values,
                left=0,
                right=0,
            )
            for w in wind_farms.index
        ]
        for dt in tqdm(wind_speed_data.index)
    ]

    df = pd.DataFrame(
        data=wind_power_data, index=wind_speed_data.index, columns=wind_farms.index
    ).round(2)
    return df


def extract_data(points, start_dt, end_dt, directory, hours_forecasted, SELECTORS):
    """Read wind speed from previously-downloaded files, and interpolate any gaps."""
    wind_data_lat_long = get_wind_data_lat_long(start_dt, directory)
    wind_farm_to_closest_wind_grid_indices = find_closest_wind_grids(
        points, wind_data_lat_long
    )
    dts = pd.date_range(start=start_dt, end=end_dt, freq="15T")
    data = {
        SELK: pd.DataFrame(index=dts, columns=points.index, dtype=float)
        for SELK in SELECTORS
    }

    fns = pd.date_range(start=start_dt, end=end_dt, freq="1H")
    for dt in fns:
        fn = os.path.join(
            directory, formatted_filename(
                dt, hours_forecasted=hours_forecasted)
        )
        print(f"Reading {fn}")
        try:
            gribs = pygrib.open(fn)
            for SELK, SELV in SELECTORS.items():
                try:
                    for u in gribs.select(name=SELV):
                        data_key = dt + \
                            datetime.timedelta(minutes=int(u.validityTime))
                        data[SELK].loc[data_key] = u.values.flatten()[
                            wind_farm_to_closest_wind_grid_indices
                        ]
                except Exception as e:
                    log_error(
                        e,
                        formatted_filename(
                            dt, hours_forecasted=hours_forecasted),
                        message="in reading the files",
                    )
                    for i in range(0, 60, 15):
                        data_key = dt + datetime.timedelta(minutes=i)
                        data[SELK].loc[data_key] = np.nan
        except Exception as e:
            log_error(
                e,
                formatted_filename(dt, hours_forecasted=hours_forecasted),
                message="in opening the files",
            )
            print(f"Grib file {dt} not found")

    for k, v in data.items():
        data[k] = linear(v.sort_index(), inplace=False)
        data[k].columns = points["Bus"]

    if "UWind80" in data.keys() and "VWind80" in data.keys():
        data["Wind80"] = np.sqrt(
            pow(data["UWind80"], 2) + pow(data["VWind80"], 2))
    if "UWind10" in data.keys() and "VWind10" in data.keys():
        data["Wind10"] = np.sqrt(
            pow(data["UWind10"], 2) + pow(data["VWind10"], 2))

    return data


def extract_data_parallel(
    points, start_dt, end_dt, directory, hours_forecasted, SELECTORS
):
    wind_data_lat_long = get_wind_data_lat_long(start_dt, directory)
    wind_farm_to_closest_wind_grid_indices = find_closest_wind_grids(
        points, wind_data_lat_long
    )
    dts = pd.date_range(start=start_dt, end=end_dt, freq="15T")
    data = {
        SELK: pd.DataFrame(index=dts, columns=points.index, dtype=float)
        for SELK in SELECTORS
    }

    fns = pd.date_range(start=start_dt, end=end_dt, freq="1H")

    def process_time_slice(dt):
        fn = os.path.join(
            directory, formatted_filename(
                dt, hours_forecasted=hours_forecasted)
        )
        print(f"Reading {fn}")
        try:
            gribs = pygrib.open(fn)
            for SELK, SELV in SELECTORS.items():
                try:
                    for u in gribs.select(name=SELV):
                        data_key = dt + \
                            datetime.timedelta(minutes=int(u.validityTime))
                        data[SELK].loc[data_key] = u.values.flatten()[
                            wind_farm_to_closest_wind_grid_indices
                        ]
                except Exception as e:
                    log_error(
                        e,
                        formatted_filename(
                            dt, hours_forecasted=hours_forecasted),
                        message="in reading the files",
                    )
                    for i in range(0, 60, 15):
                        data_key = dt + datetime.timedelta(minutes=i)
                        data[SELK].loc[data_key] = np.nan
        except Exception as e:
            log_error(
                e,
                formatted_filename(dt, hours_forecasted=hours_forecasted),
                message="in opening the files",
            )
            print(f"Grib file {dt} not found")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_time_slice, fns)

    for k, v in data.items():
        data[k] = linear(v.sort_index(), inplace=False)
        data[k].columns = points["Bus"]

    if "UWind80" in data.keys() and "VWind80" in data.keys():
        data["Wind80"] = np.sqrt(
            pow(data["UWind80"], 2) + pow(data["VWind80"], 2))
    if "UWind10" in data.keys() and "VWind10" in data.keys():
        data["Wind10"] = np.sqrt(
            pow(data["UWind10"], 2) + pow(data["VWind10"], 2))

    return data
