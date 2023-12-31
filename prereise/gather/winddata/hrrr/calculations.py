import functools
import os

import numpy as np
import pandas as pd
from powersimdata.utility.distance import ll2uv
from scipy.spatial import KDTree
from tqdm import tqdm

from prereise.gather.winddata import const
from prereise.gather.winddata.hrrr.helpers import formatted_filename
from prereise.gather.winddata.impute import linear
from prereise.gather.winddata.power_curves import (
    get_power,
    get_state_power_curves,
    get_turbine_power_curves,
    shift_turbine_curve,
)
import datetime
import pygrib

def log_error(e, filename, hours_forecasted="0", message="in something"):
    """logging error"""
    print(
        f"in {filename} ERROR: {e} occured {message} for hours forcast {hours_forecasted}"
    )
    return


def get_wind_data_lat_long(dt, directory, hours_forecasted="0"):
    """Returns the latitude and longitudes of the various
    wind grid sectors. Function assumes that there's data
    for the dt provided and the data lives in the directory.

    :param datetime.datetime dt: date and time of the grib data
    :param str directory: directory where the data is located
    :return: (*tuple*) -- A tuple of 2 same lengthed numpy arrays, first one being
        latitude and second one being longitude.
    """
    try:
        import pygrib
    except ImportError:
        print("pygrib is missing but required for this function")
        raise
    gribs = pygrib.open(
        os.path.join(
            directory, formatted_filename(dt, hours_forecasted=hours_forecasted)
        )
    )
    grib = next(gribs)
    # , grib['numberOfDataPoints']

    return grib.latlons()


def find_closest_wind_grids(wind_farms, wind_data_lat_long):
    """Uses provided wind farm data and wind grid data to calculate
    the closest wind grid to each wind farm.

    :param pandas.DataFrame wind_farms: plant data frame.
    :param tuple wind_data_lat_long: A tuple of 2 same lengthed numpy arrays, first one being
        latitude and second one being longitude.
    :return: (*numpy.array*) -- a numpy array that holds in each index i
        the index of the closest wind grid in wind_data_lat_long for wind_farms i
    """
    grid_lats, grid_lons = (
        wind_data_lat_long[0].flatten(),
        wind_data_lat_long[1].flatten(),
    )
    assert len(grid_lats) == len(grid_lons)
    grid_lat_lon_unit_vectors = [ll2uv(i, j) for i, j in zip(grid_lons, grid_lats)]

    tree = KDTree(grid_lat_lon_unit_vectors)

    wind_farm_lats = wind_farms.lat.values
    wind_farm_lons = wind_farms.lon.values

    wind_farm_unit_vectors = [
        ll2uv(i, j) for i, j in zip(wind_farm_lons, wind_farm_lats)
    ]
    _, indices = tree.query(wind_farm_unit_vectors)

    return indices


def calculate_pout_individual(
    wind_speed_data, wind_farms, start_dt, end_dt, directory  # ,  hours_forecasted
):
    """Calculate power output for wind farms based on hrrr data. Each wind farm's power
    curve is based on farm-specific attributes.
    Function assumes that user has already called
    :meth:`prereise.gather.winddata.hrrr.hrrr.retrieve_data` with the same
    ``start_dt``, ``end_dt``, and ``directory``.

    :param pandas.DataFrame wind_farms: plant data frame, plus additional columns:
        'Predominant Turbine Manufacturer', 'Predominant Turbine Model Number', and
        'Turbine Hub Height (Feet)'.
    :param str start_dt: start date.
    :param str end_dt: end date (inclusive).
    :param str directory: directory where hrrr data is contained.
    :return: (*pandas.Dataframe*) -- data frame containing power out per wind
        farm on a per hourly basis between start_dt and end_dt inclusive. Structure of
        dataframe is:
            wind_farm1  wind_farm2
        dt1    POUT        POUT
        dt2    POUT        POUT
    :raises ValueError: if ``wind_farms`` is missing the 'state_abv' column.
    """

    def get_starting_curve_name(series, valid_names):
        """Given a wind farm series, build a single string used to look up a wind farm
        power curve. If the specific make and model aren't found, return a default.

        :param pandas.Series series: single row of wind farm table.
        :param iterable valid_names: set of valid names.
        :return: (*str*) -- valid lookup name.
        """
        full_name = " ".join([series[const.mfg_col], series[const.model_col]])
        return full_name if full_name in valid_names else "IEC class 2"

    def get_shifted_curve(lookup_name, hub_height, reference_curves):
        """Get the power curve for the given turbine type, shifted to hub height.

        :param str lookup_name: make and model of turbine (or IEC class).
        :param int/float hub_height: turbine hub height.
        :param dict/pandas.DataFrame reference_curves: turbine power curves.
        :return: (*pandas.Series*) -- index is wind speed at 80m, values are normalized
        power output.
        """
        return shift_turbine_curve(
            reference_curves[lookup_name],
            hub_height,
            const.max_wind_speed,
            const.new_curve_res,
        )

    def interpolate(wspd, curve):
        return np.interp(wspd, curve.index.values, curve.values, left=0, right=0)

    req_cols = {const.mfg_col, const.model_col, const.hub_height_col}
    if not req_cols <= set(wind_farms.columns):
        raise ValueError(f"wind_farms requires columns: {req_cols}")

    # Create cached, curried function for use with apply
    turbine_power_curves = get_turbine_power_curves()
    cached_func = functools.lru_cache(maxsize=None)(
        functools.partial(get_shifted_curve, reference_curves=turbine_power_curves)
    )
    # Create dataframe of lookup values
    lookup_names = wind_farms.apply(
        lambda x: get_starting_curve_name(x, turbine_power_curves.columns), axis=1
    )
    lookup_values = pd.concat([lookup_names, wind_farms[const.hub_height_col]], axis=1)
    # Use lookup values with cached, curried function
    shifted_power_curves = lookup_values.apply(lambda x: cached_func(*x), axis=1)

    dts = wind_speed_data.index
    wind_power_data = [
        [
            interpolate(wind_speed_data.loc[dt, w], shifted_power_curves.loc[w])
            for w in wind_farms.index
        ]
        for dt in tqdm(dts)
    ]

    df = pd.DataFrame(data=wind_power_data, index=dts, columns=wind_farms.index).round(
        2
    )

    return df


def extract_data(
    points, start_dt, end_dt, directory, hours_forecasted, SELECTORS
):
    """Read wind speed from previously-downloaded files, and interpolate any gaps.

    :param pandas.DataFrame wind_farms: plant data frame.
    :param str start_dt: start date.
    :param str end_dt: end date (inclusive).
    :param str directory: directory where hrrr data is contained.
    :return: (*pandas.Dataframe*) -- data frame containing wind speed per wind farm
        on a per hourly basis between start_dt and end_dt inclusive. Structure of
        dataframe is:
            wind_farm1  wind_farm2
        dt1   speed       speed
        dt2   speed       speed
    """
    import datetime

    try:
        import pygrib
    except ImportError:
        print("pygrib is missing but required for this function")
        raise
    wind_data_lat_long = get_wind_data_lat_long(start_dt, directory)
    wind_farm_to_closest_wind_grid_indices = find_closest_wind_grids(
        points, wind_data_lat_long
    )
    dts = pd.date_range(start=start_dt, end=end_dt, freq="15T").to_pydatetime()
    # cols = list(range(numberOfDataPoints))
    cols = points.index

    # df = pd.DataFrame( index=dts, columns=cols, dtype=float,)
    data = {}
    for SELK, SELV in SELECTORS.items():
        data[SELK] = pd.DataFrame(
            index=dts,
            columns=cols,
            dtype=float,
        )
    fns = pd.date_range(start=start_dt, end=end_dt, freq="1H").to_pydatetime()
    # for dt in tqdm(fns):
    for dt in fns:
        fn = os.path.join(
            directory, formatted_filename(dt, hours_forecasted=hours_forecasted)
        )
        print(f"reading {fn}")
        try:
            gribs = pygrib.open(fn)
            for SELK, SELV in SELECTORS.items():
                try:
                    # now that the file is open, read the data for wind/solar/temp
                    for ux, u in enumerate(gribs.select(name=SELV)):
                        if "(instant)" in str(u):
                            # print(SELK, SELV)
                            # print(u)
                            # NOTE XXX added a space before 0 so it doesn't get confused with 30 mins
                            if (
                                " 0 mins" in str(u)
                                or " 1 mins" in str(u)
                                or "60 mins" in str(u)
                            ):
                                data[SELK].loc[
                                    dt + datetime.timedelta(minutes=0)
                                ] = u.values.flatten()[
                                    wind_farm_to_closest_wind_grid_indices
                                ]
                            elif "15 mins" in str(u):
                                data[SELK].loc[
                                    dt + datetime.timedelta(minutes=15)
                                ] = u.values.flatten()[
                                    wind_farm_to_closest_wind_grid_indices
                                ]
                            elif "30 mins" in str(u):
                                data[SELK].loc[
                                    dt + datetime.timedelta(minutes=30)
                                ] = u.values.flatten()[
                                    wind_farm_to_closest_wind_grid_indices
                                ]
                            elif "45 mins" in str(u):
                                data[SELK].loc[
                                    dt + datetime.timedelta(minutes=45)
                                ] = u.values.flatten()[
                                    wind_farm_to_closest_wind_grid_indices
                                ]
                            else:
                                print(
                                    "one of the values for 0,15, 30, and 45 is not found"
                                )
                except Exception as e:
                    log_error(
                        e,
                        formatted_filename(dt, hours_forecasted=hours_forecasted),
                        message="in reading the files",
                    )
                    # I don't think we need this; if the data is missing, we just skip it
                    # we don't need to put it at nan; why?
                    data[SELK].loc[dt + datetime.timedelta(minutes=0)] = np.nan
                    data[SELK].loc[dt + datetime.timedelta(minutes=15)] = np.nan
                    data[SELK].loc[dt + datetime.timedelta(minutes=30)] = np.nan
                    data[SELK].loc[dt + datetime.timedelta(minutes=45)] = np.nan

                # gribs.close()
        except Exception as e:
            log_error(
                e,
                formatted_filename(dt, hours_forecasted=hours_forecasted),
                message="in openning of the files",
            )
            print(f"grib file {dt} not found")
            # close the grib file; maybe this helps to make it run faster

    for k, v in data.items():
        data[k] = linear(v.sort_index(), inplace=False)
        data[k].columns = points["Bus"]

    # calculate total wind sqrt(u-component^2 + v-component^2), do this only for the data that contains wind
    try:
        if "UWind80" in data.keys() and "VWind80" in data.keys():
            data["Wind80"] = np.sqrt(pow(data["UWind80"], 2) + pow(data["VWind80"], 2))
    except Exception as e:
        print(e)
        raise
    try:
        if "UWind10" in data.keys() and "VWind10" in data.keys():
            data["Wind10"] = np.sqrt(pow(data["UWind10"], 2) + pow(data["VWind10"], 2))
    except Exception as e:
        print(e)
        raise

    return data


import concurrent.futures

def extract_data_parallel(points, start_dt, end_dt, directory, hours_forecasted, SELECTORS):
    wind_data_lat_long = get_wind_data_lat_long(start_dt, directory)
    wind_farm_to_closest_wind_grid_indices = find_closest_wind_grids(points, wind_data_lat_long)
    dts = pd.date_range(start=start_dt, end=end_dt, freq="15T")
    data = {SELK: pd.DataFrame(index=dts, columns=points.index, dtype=float) for SELK in SELECTORS}

    fns = pd.date_range(start=start_dt, end=end_dt, freq="1H")
    
    def process_time_slice(dt):
        fn = os.path.join(directory, formatted_filename(dt, hours_forecasted=hours_forecasted))
        print(f"Reading {fn}")
        try:
            gribs = pygrib.open(fn)
            for SELK, SELV in SELECTORS.items():
                try:
                    for u in gribs.select(name=SELV):
                        data_key = dt + datetime.timedelta(minutes=int(u.validityTime))
                        data[SELK].loc[data_key] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                except Exception as e:
                    log_error(e, formatted_filename(dt, hours_forecasted=hours_forecasted), message="in reading the files")
                    for i in range(0, 60, 15):
                        data_key = dt + datetime.timedelta(minutes=i)
                        data[SELK].loc[data_key] = np.nan
        except Exception as e:
            log_error(e, formatted_filename(dt, hours_forecasted=hours_forecasted), message="in opening the files")
            print(f"Grib file {dt} not found")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_time_slice, fns)
    
    for k, v in data.items():
        data[k] = linear(v.sort_index(), inplace=False)
        data[k].columns = points["Bus"]

    if "UWind80" in data.keys() and "VWind80" in data.keys():
        data["Wind80"] = np.sqrt(pow(data["UWind80"], 2) + pow(data["VWind80"], 2))
    if "UWind10" in data.keys() and "VWind10" in data.keys():
        data["Wind10"] = np.sqrt(pow(data["UWind10"], 2) + pow(data["VWind10"], 2))

    return data

