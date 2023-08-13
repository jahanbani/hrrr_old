""" For this code to work you need to change the core.py of Herbie
The file names must be changed. search for localfilename
However, if we don't change this it seems that there are 
two files that are being downloaded, why?
It is located in /local/alij/anaconda3/envs/herbie/lib/python3.11/site-packages/herbie
"""
import concurrent.futures
from datetime import datetime

# I need to calculate the solar output here
import ipdb
import numpy as np
import time
import pandas as pd
import PySAM.Pvwattsv8 as PVWatts
import PySAM.PySSC as pssc  # noqa: N813
import PySAM.ResourceTools as RT
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Bing, Photon
from herbie import Herbie
from herbie.fast import FastHerbie
from IPython import embed
from powersimdata.input.grid import Grid
from prereise.gather.const import abv2state, SELECTORS
from prereise.gather.solardata.helpers import to_reise
from prereise.gather.solardata.nsrdb import naive
from prereise.gather.solardata.nsrdb.sam import (
    retrieve_data_blended,
    retrieve_data_individual,
    retrieve_data_individual_ali,
    retrieve_data_individual_orig,
)
from prereise.gather.winddata.hrrr.calculations import (
    calculate_pout_individual,
    extract_solar_data,
    extract_wind_speed,
)
from prereise.gather.winddata.hrrr.hrrr import retrieve_data
from tqdm import tqdm

# two things we do here
# 1. download the data
# 2. calculate output power of wind/solar
# 3. XXX look at the linear funciton; it is used for wind speed why not for other solar params


geolocator = Photon(user_agent="geoapiExercises")


def calculate_solar_power(solar_data, pv_dict):
    """Use PVWatts to translate weather data into power.

    :param dict solar_data: weather data as returned by :meth:`Psm3Data.to_dict`.
    :param dict pv_dict: solar plant attributes.
    :return: (*numpy.array*) hourly power output.
    """
    pv_dat = pssc.dict_to_ssc_table(pv_dict, "pvwattsv8")
    pv = PVWatts.wrap(pv_dat)
    pv.SolarResource.assign({"solar_resource_data": solar_data})
    pv.execute()
    return np.array(pv.Outputs.gen)


def get_states(df):
    """takes lat lon and returns state
    XXX this seems to not work XXX FIXME
    """
    # Replace YOUR_API_KEY with your own Bing Maps API key
    geolocator = Bing(
        api_key="Aqmzwd34w1kATW-28eBQ1vRrdW3p8A2xAqeo3uokayav_LQqN-LqCrDPBtP_YSoM"
    )
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    def get_location_by_coordinates(lat, lon):
        location = geocode((lat, lon))
        return location.raw["address"].get("adminDistrict", "")

    def process_df(df):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                states = list(
                    executor.map(get_location_by_coordinates,
                                 df["lat"], df["lon"])
                )
                df["state"] = states
            except Exception as e:
                print(e)
                df["state"] = "offshore"
        return df

    # read the In_windsolarlocations.xlsx
    # df = pd.read_excel("In_windsolarlocations.xlsx")[["lat", "lon"]]
    df = process_df(df)

    return df


def prepare_solar(solar_plantx):
    """prepare the solar data to be used in the power calculation"""

    t1 = time.time()
    print('adding states to solar plant')
    # df = get_states(solar_plantx)
    solar_plantx['state'] = 'offshore'
    df = solar_plantx
    print('states added')
    t2 = time.time()
    print(f'it took {t2-t1} seconds to add states')

    solar_plant = pd.merge(
        df,
        pd.DataFrame({"state": abv2state.values(),
                     "state_abv": abv2state.keys()}),
        on="state",
        how="left",
    ).rename(columns={"state": "interconnect"})
    # XXX there are some lat/long outside of the US, see bus: 601038, I'll put MN instead (closest)
    solar_plant["state_abv"] = solar_plant["state_abv"].fillna(value="MN")
    solar_plant["zone_id"] = 1
    solar_plant.loc[:, "Fixed Tilt?"] = False
    solar_plant.loc[:, "Single-Axis Tracking?"] = True
    solar_plant.loc[:, "Dual-Axis Tracking?"] = False
    solar_plant.index.name = "plant_id"

    return solar_plant


def prepare_wind(wind_farms):
    # wind_farms = pd.read_excel("In_windsolarlocations.xlsx")
    wind_farms.loc[
        wind_farms["Offshore"] == 0,
        ["Predominant Turbine Model Number", "Predominant Turbine Manufacturer"],
    ] = "IEC class 2"
    wind_farms.loc[:, "Turbine Hub Height (Feet)"] = 328
    # if offshore, set tthe Predominant Turbine Model Number to V236 and Predominant Turbine Manufacturer to Vestas
    wind_farms.loc[
        wind_farms["Offshore"] == 1, "Predominant Turbine Model Number"
    ] = "V236"
    wind_farms.loc[
        wind_farms["Offshore"] == 1, "Predominant Turbine Manufacturer"
    ] = "Vestas"
    return wind_farms


OUTDIR = "../psse/grg-pssedata/"

# output of wind turbines at 100 m
WIND_POWER_OUTPUT_FILE = OUTDIR + "wind_power"
WIND_SPEED_FN = OUTDIR + "wind_speed"
SOLAR_WIND_SPEED_FN = OUTDIR + "solar_wind_speed"
SOLAR_TMP_DATA_FN = OUTDIR + "solar_tmp_data"
SOLAR_RAD_DATA_FN = OUTDIR + "solar_rad_data"
SOLAR_VDD_DATA_FN = OUTDIR + "solar_vdd_data"
SOLAR_VBD_DATA_FN = OUTDIR + "solar_vbd_data"
SOLAR_OUTPUT_FILE = OUTDIR + "solar_power"

# study year; which year to study
YEAR = 2020
DIR = r"/research/alij/"
START = pd.to_datetime("2019-12-30 00:00")
END = pd.to_datetime("2021-01-05 01:00")

# retrieve_data(START, END, DIR)

# download the data
DF = 0  # download Flag
if DF:
    print("prepare to download")
    # Create a range of dates
    FHDATES = pd.date_range(
        # start="2019-12-30 00:00",
        # end="2021-01-05 01:00",
        start=START,
        end=END,
        freq="1H",
    )

    print("Create a range of forecast lead times")
    fxx = [0, 1]
    FH = FastHerbie(
        FHDATES,
        model="hrrr",
        fxx=fxx,
        product="subh",
        save_dir=DIR,
    )
    print("downloading")

    FH.download(
        # wind and everything for solar
        searchString="V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m",
        # wind only
        # searchString="(?:U|V)GRD:80 m",
        # solar only
        # searchString="V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:10 m",
        save_dir=DIR,
        max_threads=200,
        verbose=True,
    )


# wind
# offshore project data
colslatlons = ["Bus", "lat", "lon", "Wind", "Solar", "Offshore"]
fn = "../psse/grg-pssedata/Buses_with_lat_lon_for_visualization.csv"
fn = "../psse/grg-pssedata/Buses_with_lat_lon_for_visualization_wholeEI.csv"
latlons = pd.read_csv(fn)[colslatlons]

latlons = latlons.loc[(~(latlons["lat"].isna()) & ~(latlons["lon"].isna())), :]
wind_farms = latlons.loc[latlons["Wind"] == 1, :]
solar_plantx = latlons.loc[latlons["Solar"] == 1, :]


# calculate output power
wind_speed_data = {}
solar_wind_speed_data = {}
solar_tmp_data = {}
solar_rad_data = {}
solar_vbd_data = {}
solar_vdd_data = {}

WindRead = 0
ReadData = 0
if WindRead:
    for inx, DEFAULT_HOURS_FORECASTED in enumerate(["0", "1"]):
        # for inx, DEFAULT_HOURS_FORECASTED in enumerate(["1"]):
        print("*" * 180)
        print(f"for wind power {DEFAULT_HOURS_FORECASTED}")
        print("*" * 180)

        # Read wind speed from previously-downloaded files, and impute as necessary
        (
            wind_speed_data_inx,
            solar_wind_speed_data_inx,
            solar_tmp_data_inx,
            solar_rad_data_inx,
            solar_vbd_data_inx,
            solar_vdd_data_inx,
        ) = extract_wind_speed(
            wind_farms, solar_plantx, START, END, DIR, DEFAULT_HOURS_FORECASTED, SELECTORS
        )
        wind_speed_data[inx] = wind_speed_data_inx
        solar_wind_speed_data[inx] = solar_wind_speed_data_inx
        solar_tmp_data[inx] = solar_tmp_data_inx
        solar_rad_data[inx] = solar_rad_data_inx
        solar_vbd_data[inx] = solar_vbd_data_inx
        solar_vdd_data[inx] = solar_vdd_data_inx

    wind_speed_data[1].loc[wind_speed_data[1].index.minute == 0, :] = wind_speed_data[
        0
    ].loc[wind_speed_data[0].index.minute == 0, :]
    solar_wind_speed_data[1].loc[
        solar_wind_speed_data[1].index.minute == 0, :
    ] = solar_wind_speed_data[0].loc[solar_wind_speed_data[0].index.minute == 0, :]
    solar_tmp_data[1].loc[solar_tmp_data[1].index.minute == 0, :] = solar_tmp_data[
        0
    ].loc[solar_tmp_data[0].index.minute == 0, :]
    solar_rad_data[1].loc[solar_rad_data[1].index.minute == 0, :] = solar_rad_data[
        0
    ].loc[solar_rad_data[0].index.minute == 0, :]
    solar_vbd_data[1].loc[solar_vbd_data[1].index.minute == 0, :] = solar_vbd_data[
        0
    ].loc[solar_vbd_data[0].index.minute == 0, :]
    solar_vdd_data[1].loc[solar_vdd_data[1].index.minute == 0, :] = solar_vdd_data[
        0
    ].loc[solar_vdd_data[0].index.minute == 0, :]

    wind_speed_data[1].to_excel(WIND_SPEED_FN + ".xlsx")
    solar_wind_speed_data[1].to_excel(SOLAR_WIND_SPEED_FN + ".xlsx")
    solar_tmp_data[1].to_excel(SOLAR_TMP_DATA_FN + ".xlsx")
    solar_rad_data[1].to_excel(SOLAR_RAD_DATA_FN + ".xlsx")
    solar_vbd_data[1].to_excel(SOLAR_VBD_DATA_FN + ".xlsx")
    solar_vdd_data[1].to_excel(SOLAR_VDD_DATA_FN + ".xlsx")

    # to parquet
    # Covnvert all column names to string
    wind_speed_data[1].columns = wind_speed_data[1].columns.astype(str)
    solar_wind_speed_data[1].columns = solar_wind_speed_data[1].columns.astype(
        str)
    solar_tmp_data[1].columns = solar_tmp_data[1].columns.astype(str)
    solar_rad_data[1].columns = solar_rad_data[1].columns.astype(str)
    solar_vbd_data[1].columns = solar_vbd_data[1].columns.astype(str)
    solar_vdd_data[1].columns = solar_vdd_data[1].columns.astype(str)

    # write the data in parquet format
    wind_speed_data[1].to_parquet(WIND_SPEED_FN + ".parquet")
    solar_wind_speed_data[1].to_parquet(SOLAR_WIND_SPEED_FN + ".parquet")
    solar_tmp_data[1].to_parquet(SOLAR_TMP_DATA_FN + ".parquet")
    solar_rad_data[1].to_parquet(SOLAR_RAD_DATA_FN + ".parquet")
    solar_vbd_data[1].to_parquet(SOLAR_VBD_DATA_FN + ".parquet")
    solar_vdd_data[1].to_parquet(SOLAR_VDD_DATA_FN + ".parquet")

    wind_speed_data = wind_speed_data[1]
    solar_wind_speed_data = solar_wind_speed_data[1]
    solar_tmp_data = solar_tmp_data[1]
    solar_rad_data = solar_rad_data[1]
    solar_vbd_data = solar_vbd_data[1]
    solar_vdd_data = solar_vdd_data[1]
elif ReadData:
    print(f"reading {WIND_SPEED_FN}")
    # wind_speed_data = pd.read_excel(WIND_SPEED_FN+".xlsx", index_col=0)
    wind_speed_data = pd.read_parquet(WIND_SPEED_FN + ".parquet")
    # convert the column names back to integers
    wind_speed_data.columns = wind_speed_data.columns.astype(int)
    wind_speed_data[wind_speed_data < 0] = 0

    print(f"reading {SOLAR_WIND_SPEED_FN}")
    # solar_wind_speed_data = pd.read_excel(SOLAR_WIND_SPEED_FN+".xlsx", index_col=0)
    solar_wind_speed_data = pd.read_parquet(SOLAR_WIND_SPEED_FN + ".parquet")
    # convert the column names back to integers
    solar_wind_speed_data.columns = solar_wind_speed_data.columns.astype(int)
    solar_wind_speed_data[solar_wind_speed_data < 0] = 0

    print(f"reading {SOLAR_TMP_DATA_FN}")
    # solar_tmp_data = pd.read_excel(SOLAR_TMP_DATA_FN+".xlsx", index_col=0)
    solar_tmp_data = pd.read_parquet(SOLAR_TMP_DATA_FN + ".parquet")
    solar_tmp_data[solar_tmp_data < 0] = 0
    # convert the column names back to integers
    solar_tmp_data.columns = solar_tmp_data.columns.astype(int)

    print(f"reading {SOLAR_RAD_DATA_FN}")
    # solar_rad_data = pd.read_excel(SOLAR_RAD_DATA_FN+".xlsx", index_col=0)
    solar_rad_data = pd.read_parquet(SOLAR_RAD_DATA_FN + ".parquet")
    # convert the column names back to integers
    solar_rad_data.columns = solar_rad_data.columns.astype(int)
    solar_rad_data[solar_rad_data < 0] = 0
    print(f"reading {SOLAR_VDD_DATA_FN}")
    # solar_vdd_data = pd.read_excel(SOLAR_VDD_DATA_FN+".xlsx", index_col=0)
    solar_vdd_data = pd.read_parquet(SOLAR_VDD_DATA_FN + ".parquet")
    # convert the column names back to integers
    solar_vdd_data.columns = solar_vdd_data.columns.astype(int)
    solar_vdd_data[solar_vdd_data < 0] = 0
    print(f"reading {SOLAR_VBD_DATA_FN}")
    # solar_vbd_data = pd.read_excel(SOLAR_VBD_DATA_FN+".xlsx", index_col=0)
    solar_vbd_data = pd.read_parquet(SOLAR_VBD_DATA_FN + ".parquet")
    # convert the column names back to integers
    solar_vbd_data.columns = solar_vbd_data.columns.astype(int)
    solar_vbd_data[solar_vbd_data < 0] = 0


print("preparing wind farms")
wind_farms = prepare_wind(wind_farms)
WindCalc = 0
if WindCalc:
    # shift wind speed data to Eastern time
    # XXX we can do it this way or the .shift(5) way (like solar)
    wind_speed_data.index = wind_speed_data.index - pd.Timedelta(hours=5)

    wind_power = calculate_pout_individual(
        wind_speed_data,
        wind_farms,
        start_dt=START,
        end_dt=END,
        directory=DIR,
        # hours_forecasted=DEFAULT_HOURS_FORECASTED,
    )

    wind_power.to_excel(WIND_POWER_OUTPUT_FILE + ".xlsx")
    # convert the columns of wind power to string
    wind_power.columns = wind_power.columns.astype(str)
    wind_power.to_parquet(WIND_POWER_OUTPUT_FILE + ".parquet")


print("preparing solar plant")
solar_plant = prepare_solar(solar_plantx)
print("preparing solar")


def prepare_calculate_solar_power(
    solar_wind_speed_data, solar_tmp_data, solar_vdd_data, solar_vbd_data, solar_plant
):
    solar_data_all = {
        "wspd": solar_wind_speed_data,
        "df": solar_vdd_data,
        "dn": solar_vbd_data,
        "tdry": solar_tmp_data - 273.15,
    }
    df = pd.concat(solar_data_all.values(), keys=solar_data_all.keys())

    latlon = solar_plant.loc[:, ["lat", "lon"]]

    dfsa = {}
    for inx, (lat, lon) in enumerate(latlon.values):
        print(f"for index {inx}")
        dfs = df[[inx]].unstack(0)[inx]

        # shift 6 hours UTC to CST XXX NOTE THIS needs to be changed for EST
        dfs = dfs.shift(periods=-5)
        dfs.loc[:, "year"] = dfs.index.year
        dfs.loc[:, "month"] = dfs.index.month
        dfs.loc[:, "day"] = dfs.index.day
        dfs.loc[:, "hour"] = dfs.index.hour
        dfs.loc[:, "minute"] = dfs.index.minute
        dfs = dfs.loc[dfs.index.year == YEAR, :]
        dfs.loc[dfs["df"] > 1000, "df"] = 1000
        dfs.loc[dfs["dn"] > 1000, "dn"] = 1000
        dfs = dfs.reset_index(drop=True).fillna(method="ffill")
        dfsd = dfs.to_dict(orient="list")
        dfsd["lat"] = lat
        dfsd["lon"] = lon
        dfsd["tz"] = -5
        dfsd["elev"] = 898

        dfsa[inx] = dfsd
        if dfs.isnull().values.any():
            print(f"{lat} and {lon} at {inx} has NaN")

    default_pv_parameters = {
        "adjust:constant": 0,
        "azimuth": 180,
        "gcr": 0.4,
        "inv_eff": 94,
        "losses": 14,
        "tilt": 30,
    }
    ilr = 1.25

    plant_pv_dict = {
        "system_capacity": ilr,
        "dc_ac_ratio": ilr,
        "array_type": 1,
    }

    pv_dict = {**default_pv_parameters, **plant_pv_dict}

    dff = {}
    for k, dfd in dfsa.items():
        print(f"calculate power for solar plant number {k}")
        power = calculate_solar_power(dfd, pv_dict)
        dff[k] = pd.DataFrame(power).rename(columns={0: k})
        # print(dff[k].loc[dff[k][k] > 0])
        if dff[k].loc[dff[k][k] > 0].empty:
            print(f"output of {k} is all zeros")
        if dff[k].loc[dff[k][k] > 0].max().values > 1:
            print(f"output of {k} is greater than zero")

    dfl = []
    for k, df in dff.items():
        dfl.append(df)

    dfindex = pd.date_range(
        start=str(YEAR) + "-01-01 00:00",
        end=str(YEAR + 1) + "-01-01 00:00",
        freq="15T",
        inclusive="left",
    )

    leapdfindex = dfindex[((dfindex.month == 2) & (dfindex.day == 29))]
    dfindex = dfindex[~((dfindex.month == 2) & (dfindex.day == 29))]
    df = pd.concat(dfl, axis=1)
    df.index = dfindex

    # 2020 is leap year; add data for 29th of Feb.
    if YEAR == 2020:
        leapdata = df.loc[((df.index.month == 2) & (df.index.day == 28))]
        leapdata.index = leapdfindex
        df = pd.concat([df, leapdata], axis=0).sort_index(axis=0)

    # df.round(3).to_csv( SOLAR_OUTPUT_FILE,)
    # convert the columns to strings
    df.columns = df.columns.astype(str)
    df.to_parquet(SOLAR_OUTPUT_FILE + ".parquet")
    return df


SOLARCALC = False
if SOLARCALC:
    prepare_calculate_solar_power(
        solar_wind_speed_data,
        solar_tmp_data,
        solar_vdd_data,
        solar_vbd_data,
        solar_plant,
    )


# calculate capacity factor for one year for each block
# read hours for each block from the constant file

print("read solar data")
solar_power = pd.read_parquet(SOLAR_OUTPUT_FILE + ".parquet")
# convert the index to datetime format
solar_power.index = pd.to_datetime(solar_power.index)
print("read wind data")
# wind_power = pd.read_excel(WIND_POWER_OUTPUT_FILE, index_col=0)
wind_power = pd.read_parquet(WIND_POWER_OUTPUT_FILE + ".parquet")
# convert the index to datetime format
wind_power.index = pd.to_datetime(wind_power.index)
# only select the rows where the year of the index is YEAR
wind_power = wind_power.loc[wind_power.index.year == YEAR]

__import__("ipdb").set_trace()
