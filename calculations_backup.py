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

# wind for wind farms
U_COMPONENT_SELECTOR = "U component of wind"
V_COMPONENT_SELECTOR = "V component of wind"

# wind for solar plants
SU_COMPONENT_SELECTOR = "10 metre U wind component"
SV_COMPONENT_SELECTOR = "10 metre V wind component"

# solar radiation
SR_COMPONENT_SELECTOR = "Downward short-wave radiation flux"

SB_COMPONENT_SELECTOR = "Visible Beam Downward Solar Flux"
SD_COMPONENT_SELECTOR = "Visible Diffuse Downward Solar Flux"

# solar temperature
TP_COMPONENT_SELECTOR = "2 metre temperature"


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
            directory, formatted_filename(
                dt, hours_forecasted=hours_forecasted)
        )
    )
    grib = next(gribs)

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
    grid_lat_lon_unit_vectors = [ll2uv(i, j)
                                 for i, j in zip(grid_lons, grid_lats)]

    tree = KDTree(grid_lat_lon_unit_vectors)

    wind_farm_lats = wind_farms.lat.values
    wind_farm_lons = wind_farms.lon.values

    wind_farm_unit_vectors = [
        ll2uv(i, j) for i, j in zip(wind_farm_lons, wind_farm_lats)
    ]
    _, indices = tree.query(wind_farm_unit_vectors)

    return indices

def calculate_pout_individual(
    wind_farms, start_dt, end_dt, directory, hours_forecasted
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
        functools.partial(get_shifted_curve,
                          reference_curves=turbine_power_curves)
    )
    # Create dataframe of lookup values
    lookup_names = wind_farms.apply(
        lambda x: get_starting_curve_name(x, turbine_power_curves.columns), axis=1
    )
    lookup_values = pd.concat(
        [lookup_names, wind_farms[const.hub_height_col]], axis=1)
    # Use lookup values with cached, curried function
    shifted_power_curves = lookup_values.apply(
        lambda x: cached_func(*x), axis=1)

    # Read wind speed from previously-downloaded files, and impute as necessary
    wind_speed_data = extract_wind_speed(
        wind_farms, start_dt, end_dt, directory, hours_forecasted
    )
    wind_speed_data.to_excel("wind_speed_data.xlsx")
    dts = wind_speed_data.index

    wind_power_data = [
        [
            interpolate(wind_speed_data.loc[dt, w],
                        shifted_power_curves.loc[w])
            for w in wind_farms.index
        ]
        for dt in tqdm(dts)
    ]

    df = pd.DataFrame(data=wind_power_data, index=dts,
                      columns=wind_farms.index).round(2)

    return df


def extract_wind_speed(wind_farms, start_dt, end_dt, directory, hours_forecasted):
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
        wind_farms, wind_data_lat_long
    )
    dts = pd.date_range(start=start_dt, end=end_dt, freq="15T").to_pydatetime()

    # Fetch wind speed data for each wind farm (or store NaN as applicable)
    wind_speed_data = pd.DataFrame(
        index=dts, columns=wind_farms.index, dtype=float)
    wind_speed_data_u = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    wind_speed_data_v = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    wind_speed_data = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    for dt in tqdm(dts):
        try:
            gribs = pygrib.open(
                os.path.join(
                    directory, formatted_filename(
                        dt, hours_forecasted=hours_forecasted)
                )
            )
            try:
                # now that the file is open, read the data for wind/solar/temp
                for ux, u in enumerate(gribs.select(name=U_COMPONENT_SELECTOR)):
                    if "0 mins" in str(u):
                        wind_speed_data_u.loc[
                            dt + datetime.timedelta(minutes=0)
                        # ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                        ] = u.values.flatten()
                    if "15 mins" in str(u):
                        wind_speed_data_u.loc[
                            dt + datetime.timedelta(minutes=15)
                        ] = u.values.flatten()
                    if "30 mins" in str(u):
                        wind_speed_data_u.loc[
                            dt + datetime.timedelta(minutes=30)
                        ] = u.values.flatten()
                    if "45 mins" in str(u):
                        wind_speed_data_u.loc[
                            dt + datetime.timedelta(minutes=45)
                        ] = u.values.flatten()

                for vx, v in enumerate(gribs.select(name=V_COMPONENT_SELECTOR)):
                    if "0 mins" in str(v):
                        wind_speed_data_v.loc[
                            dt + datetime.timedelta(minutes=0)
                        ] = v.values.flatten()
                    if "15 mins" in str(v):
                        wind_speed_data_v.loc[
                            dt + datetime.timedelta(minutes=15)
                        ] = v.values.flatten()
                    if "30 mins" in str(v):
                        wind_speed_data_v.loc[
                            dt + datetime.timedelta(minutes=30)
                        ] = v.values.flatten()
                    if "45 mins" in str(v):
                        wind_speed_data_v.loc[
                            dt + datetime.timedelta(minutes=45)
                        ] = v.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                # close the grib file; maybe this helps to make it run faster
                gribs.close()
            except:
                wind_speed_data_u.loc[dt + datetime.timedelta(minutes=0)] = np.nan
                wind_speed_data_u.loc[dt + datetime.timedelta(minutes=15)] = np.nan
                wind_speed_data_u.loc[dt + datetime.timedelta(minutes=30)] = np.nan
                wind_speed_data_u.loc[dt + datetime.timedelta(minutes=45)] = np.nan
                wind_speed_data_v.loc[dt + datetime.timedelta(minutes=0)] = np.nan
                wind_speed_data_v.loc[dt + datetime.timedelta(minutes=15)] = np.nan
                wind_speed_data_v.loc[dt + datetime.timedelta(minutes=30)] = np.nan
                wind_speed_data_v.loc[dt + datetime.timedelta(minutes=45)] = np.nan

        except:
            print('grib file not found')


    wind_speed_data_u = linear(wind_speed_data_u.sort_index(), inplace=False)
    wind_speed_data_v = linear(wind_speed_data_v.sort_index(), inplace=False)
    try:
        wind_speed_data = np.sqrt(
            pow(wind_speed_data_u, 2) + pow(wind_speed_data_v, 2))
    except:
        raise

    return wind_speed_data



def extract_solar_data(wind_farms, start_dt, end_dt, directory, hours_forecasted):
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
        wind_farms, wind_data_lat_long
    )
    dts = pd.date_range(start=start_dt, end=end_dt, freq="15T").to_pydatetime()

    # Fetch wind speed data for each wind farm (or store NaN as applicable)
    solar_wind_speed_data_u = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    solar_wind_speed_data_v = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    solar_wind_speed_data = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    solar_tmp_data = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    # radiation could be horizontal or normal; but I am going to assume this is normal XXX FIXME
    solar_rad_data = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    solar_vbd_data = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )
    solar_vdd_data = pd.DataFrame(
        index=dts,
        columns=wind_farms.index,
        dtype=float,
    )

    for dt in tqdm(dts):
        # radiation of sun
        try:
            gribs = pygrib.open(
                os.path.join(
                    directory, formatted_filename(
                        dt, hours_forecasted=hours_forecasted)
                )
            )
            # XXX FIXME I think I have 2 of every data; check the gribs content
            for ux, u in enumerate([gribs.select(name=SD_COMPONENT_SELECTOR)[0]]):
                # print(u)
                if "0 mins" in str(u):
                    solar_vdd_data.loc[
                        dt + datetime.timedelta(minutes=0)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "15 mins" in str(u):
                    solar_vdd_data.loc[
                        dt + datetime.timedelta(minutes=15)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "30 mins" in str(u):
                    solar_vdd_data.loc[
                        dt + datetime.timedelta(minutes=30)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "45 mins" in str(u):
                    solar_vdd_data.loc[
                        dt + datetime.timedelta(minutes=45)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
        except:
            solar_vdd_data.loc[dt + datetime.timedelta(minutes=0)] = np.nan
            solar_vdd_data.loc[dt + datetime.timedelta(minutes=15)] = np.nan
            solar_vdd_data.loc[dt + datetime.timedelta(minutes=30)] = np.nan
            solar_vdd_data.loc[dt + datetime.timedelta(minutes=45)] = np.nan

        # radiation of sun
        try:
            # XXX FIXME I think I have 2 of every data; check the gribs content
            for ux, u in enumerate([gribs.select(name=SB_COMPONENT_SELECTOR)[0]]):
                # print(u)
                if "0 mins" in str(u):
                    solar_vbd_data.loc[
                        dt + datetime.timedelta(minutes=0)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "15 mins" in str(u):
                    solar_vbd_data.loc[
                        dt + datetime.timedelta(minutes=15)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "30 mins" in str(u):
                    solar_vbd_data.loc[
                        dt + datetime.timedelta(minutes=30)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "45 mins" in str(u):
                    solar_vbd_data.loc[
                        dt + datetime.timedelta(minutes=45)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
        except:
            solar_vbd_data.loc[dt + datetime.timedelta(minutes=0)] = np.nan
            solar_vbd_data.loc[dt + datetime.timedelta(minutes=15)] = np.nan
            solar_vbd_data.loc[dt + datetime.timedelta(minutes=30)] = np.nan
            solar_vbd_data.loc[dt + datetime.timedelta(minutes=45)] = np.nan

        # radiation of sun
        try:
            # XXX FIXME I think I have 2 of every data; check the gribs content
            for ux, u in enumerate([gribs.select(name=SR_COMPONENT_SELECTOR)[0]]):
                # print(u)
                if "0 mins" in str(u):
                    solar_rad_data.loc[
                        dt + datetime.timedelta(minutes=0)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "15 mins" in str(u):
                    solar_rad_data.loc[
                        dt + datetime.timedelta(minutes=15)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "30 mins" in str(u):
                    solar_rad_data.loc[
                        dt + datetime.timedelta(minutes=30)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "45 mins" in str(u):
                    solar_rad_data.loc[
                        dt + datetime.timedelta(minutes=45)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
        except:
            solar_rad_data.loc[dt + datetime.timedelta(minutes=0)] = np.nan
            solar_rad_data.loc[dt + datetime.timedelta(minutes=15)] = np.nan
            solar_rad_data.loc[dt + datetime.timedelta(minutes=30)] = np.nan
            solar_rad_data.loc[dt + datetime.timedelta(minutes=45)] = np.nan

        try:
            # XXX FIXME I think I have 2 of every data; check the gribs content
            for ux, u in enumerate([gribs.select(name=TP_COMPONENT_SELECTOR)[0]]):
                # print(u)
                if "0 mins" in str(u):
                    solar_tmp_data.loc[
                        dt + datetime.timedelta(minutes=0)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "15 mins" in str(u):
                    solar_tmp_data.loc[
                        dt + datetime.timedelta(minutes=15)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "30 mins" in str(u):
                    solar_tmp_data.loc[
                        dt + datetime.timedelta(minutes=30)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                if "45 mins" in str(u):
                    solar_tmp_data.loc[
                        dt + datetime.timedelta(minutes=45)
                    ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
        except:
            solar_tmp_data.loc[dt + datetime.timedelta(minutes=0)] = np.nan
            solar_tmp_data.loc[dt + datetime.timedelta(minutes=15)] = np.nan
            solar_tmp_data.loc[dt + datetime.timedelta(minutes=30)] = np.nan
            solar_tmp_data.loc[dt + datetime.timedelta(minutes=45)] = np.nan

        try:
            for ux, u in enumerate([gribs.select(name=SU_COMPONENT_SELECTOR)[0]]):
                # print(u)
                if "0 mins" in str(u):
                    solar_wind_speed_data_u.loc[
                        dt + datetime.timedelta(minutes=0)
                        # ] = u.values.flatten()[wind_farm_to_closest_wind_grid_indices]
                    ] = u.values.flatten()
                if "15 mins" in str(u):
                    solar_wind_speed_data_u.loc[
                        dt + datetime.timedelta(minutes=15)
                    ] = u.values.flatten()
                if "30 mins" in str(u):
                    solar_wind_speed_data_u.loc[
                        dt + datetime.timedelta(minutes=30)
                    ] = u.values.flatten()
                if "45 mins" in str(u):
                    solar_wind_speed_data_u.loc[
                        dt + datetime.timedelta(minutes=45)
                    ] = u.values.flatten()

            for vx, v in enumerate([gribs.select(name=SV_COMPONENT_SELECTOR)[0]]):
                if "0 mins" in str(v):
                    solar_wind_speed_data_v.loc[
                        dt + datetime.timedelta(minutes=0)
                    ] = v.values.flatten()
                if "15 mins" in str(v):
                    solar_wind_speed_data_v.loc[
                        dt + datetime.timedelta(minutes=15)
                    ] = v.values.flatten()
                if "30 mins" in str(v):
                    solar_wind_speed_data_v.loc[
                        dt + datetime.timedelta(minutes=30)
                    ] = v.values.flatten()
                if "45 mins" in str(v):
                    solar_wind_speed_data_v.loc[
                        dt + datetime.timedelta(minutes=45)
                    ] = v.values.flatten()
        except ValueError:
            solar_wind_speed_data_u.loc[dt + datetime.timedelta(minutes=0)] = np.nan
            solar_wind_speed_data_u.loc[dt + datetime.timedelta(minutes=15)] = np.nan
            solar_wind_speed_data_u.loc[dt + datetime.timedelta(minutes=30)] = np.nan
            solar_wind_speed_data_u.loc[dt + datetime.timedelta(minutes=45)] = np.nan
            solar_wind_speed_data_v.loc[dt + datetime.timedelta(minutes=0)] = np.nan
            solar_wind_speed_data_v.loc[dt + datetime.timedelta(minutes=15)] = np.nan
            solar_wind_speed_data_v.loc[dt + datetime.timedelta(minutes=30)] = np.nan
            solar_wind_speed_data_v.loc[dt + datetime.timedelta(minutes=45)] = np.nan

    solar_wind_speed_data_u = linear(solar_wind_speed_data_u.sort_index(), inplace=False)
    solar_wind_speed_data_v = linear(solar_wind_speed_data_v.sort_index(), inplace=False)

    # wind 10 m data
    try:
        solar_wind_speed_data = np.sqrt(
            pow(solar_wind_speed_data_u, 2) + pow(solar_wind_speed_data_v, 2))
    except:
        raise

    # temperature 2 m data
    solar_tmp_data = linear(solar_tmp_data.sort_index(), inplace=False)
    solar_rad_data = linear(solar_rad_data.sort_index(), inplace=False)
    solar_vbd_data = linear(solar_vbd_data.sort_index(), inplace=False)
    solar_vdd_data = linear(solar_vdd_data.sort_index(), inplace=False)

    return (
        solar_wind_speed_data.round(2),
        solar_tmp_data.round(2),
        solar_rad_data.round(2),
        solar_vbd_data.round(2),
        solar_vdd_data.round(2),
    )





def calculate_pout_blended(wind_farms, start_dt, end_dt, directory):
    """Calculate power output for wind farms based on hrrr data. Each wind farm's power
    curve is based on the average power curve for that state, based on EIA data on the
    state's turbines.
    Function assumes that user has already called
    :meth:`prereise.gather.winddata.hrrr.hrrr.retrieve_data` with the same
    ``start_dt``, ``end_dt``, and ``directory``.

    :param pandas.DataFrame wind_farms: plant data frame, plus 'state_abv' column.
    :param str start_dt: start date.
    :param str end_dt: end date (inclusive).
    :param str directory: directory where hrrr data is contained.
    :return: (*pandas.Dataframe*) -- data frame containing power out per wind farm
        on a per hourly basis between start_dt and end_dt inclusive. Structure of
        dataframe is:
            wind_farm1  wind_farm2
        dt1    POUT        POUT
        dt2    POUT        POUT
    :raises ValueError: if ``wind_farms`` is missing the 'state_abv' column.
    """

    if "state_abv" not in wind_farms.columns:
        raise ValueError(
            "The wind_farms data frame must have a 'state_abv' column")
    turbine_types = wind_farms.apply(
        lambda x: "Offshore" if x["type"] == "wind_offshore" else x["state_abv"], axis=1
    )

    turbine_power_curves = get_turbine_power_curves()
    state_power_curves = get_state_power_curves()

    # Read wind speed from previously-downloaded files, and interpolate
    wind_speed_data = extract_wind_speed(
        wind_farms, start_dt, end_dt, directory)
    dts = wind_speed_data.index

    # Then calculate wind power based on wind speed
    wind_power_data = [
        [
            get_power(
                turbine_power_curves,
                state_power_curves,
                wind_speed_data.loc[dt, w],
                turbine_types.loc[w],
            )
            for w in wind_farms.index
        ]
        for dt in tqdm(dts)
    ]
    df = pd.DataFrame(data=wind_power_data, index=dts,
                      columns=wind_farms.index)

    return df


