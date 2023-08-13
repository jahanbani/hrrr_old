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
    extract_data,
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
    print("adding states to solar plant")
    # df = get_states(solar_plantx)
    solar_plantx["state"] = "offshore"
    df = solar_plantx
    print("states added")
    t2 = time.time()
    print(f"it took {t2-t1} seconds to add states")

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
    wind_farms.loc[:, "Turbine Hub Height (Feet)"] = 262.467
    # if offshore, set tthe Predominant Turbine Model Number to V236 and Predominant Turbine Manufacturer to Vestas
    wind_farms.loc[
        wind_farms["Offshore"] == 1, "Predominant Turbine Model Number"
    ] = "V236"
    wind_farms.loc[
        wind_farms["Offshore"] == 1, "Predominant Turbine Manufacturer"
    ] = "Vestas"
    return wind_farms


# OUTDIR = "../psse/grg-pssedata/"
OUTDIR = "./"

# output of wind turbines at 100 m
WIND_POWER_OUTPUT_FILE = OUTDIR + "dut_80m_wind_power"
WIND_SPEED_FN = OUTDIR + "dut_80m_wind_speed"
SOLAR_WIND_SPEED_FN = OUTDIR + "dut_solar_wind_speed"
SOLAR_TMP_DATA_FN = OUTDIR + "dut_solar_tmp_data"
SOLAR_RAD_DATA_FN = OUTDIR + "dut_solar_rad_data"
SOLAR_VDD_DATA_FN = OUTDIR + "dut_solar_vdd_data"
SOLAR_VBD_DATA_FN = OUTDIR + "dut_solar_vbd_data"
SOLAR_OUTPUT_FILE = OUTDIR + "dut_solar_power"

# study year; which year to study
YEAR = 2020
DIR = r"/research/alij/"
START = pd.to_datetime("2019-12-30 00:00")
END = pd.to_datetime("2021-01-05 01:00")

# retrieve_data(START, END, DIR)

# download the data
DF = 0  # download Flag
def DF():
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


def get_points():
    # wind
    # offshore project data
    colslatlons = ["Bus", "lat", "lon", "Wind", "Solar"]
    fn = "../psse/InputData/In_dut_80m_Iowa_Wind_Turbines.csv"
    latlons = pd.read_csv(fn)[colslatlons]

    latlons = latlons.loc[(~(latlons["lat"].isna()) & ~(latlons["lon"].isna())), :]
    wind_farms = latlons.loc[latlons["Wind"] == 1, :]
    solar_plantx = latlons.loc[latlons["Solar"] == 1, :]

    pointsnonunique = pd.concat(
        [wind_farms[["Bus", "lat", "lon"]], solar_plantx[["Bus", "lat", "lon"]]],
        ignore_index=True,
    )
    # drop repetitive points (lat and lon columns)
    points = pointsnonunique.drop_duplicates()

    return points


def WindRead():
    dataall = {}
    for inx, DEFAULT_HOURS_FORECASTED in enumerate(["0", "1"]):
        data = extract_data(
            points,
            START,
            END,
            DIR,
            DEFAULT_HOURS_FORECASTED,
            SELECTORS,
        )
        dataall[inx] = data
    # for each SELK merge the data to get the best data
    for SELK, SELV in data.items():
        dataall[1][SELK].loc[dataall[1][SELK].index.minute == 0, :] = dataall[0][
            SELK
        ].loc[dataall[0][SELK].index.minute == 0, :]
        data[SELK] = dataall[1][SELK]
        # write them in parquet format
        # first we need ot convert columns to string
        data[SELK].columns = data[SELK].columns.astype(str)
        data[SELK].to_parquet(
            SELK + ".parquet",
        )
        return data


data = WindRead()

__import__("ipdb").set_trace()
