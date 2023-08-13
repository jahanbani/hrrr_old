# two things we do here
# 1. download the data
# 2. calculate output power of wind/solar
# 3. XXX look at the linear funciton; it is used for wind speed why not for other solar params

from datetime import datetime

# I need to calculate the solar output here
import ipdb
import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as PVWatts
import PySAM.PySSC as pssc  # noqa: N813
import PySAM.ResourceTools as RT
from geopy.geocoders import Nominatim
from herbie import Herbie
from herbie.tools import FastHerbie
from IPython import embed
from powersimdata.input.grid import Grid
from tqdm import tqdm

from prereise.gather.const import abv2state
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
)
from prereise.gather.winddata.hrrr.hrrr import retrieve_data

geolocator = Nominatim(user_agent="geoapiExercises")

YEAR = 2020
DIR = r"/research/alij/All"
START = pd.to_datetime("2019-12-31 00:00")
END = pd.to_datetime("2021-01-02 01:00")

# retrieve_data(START, END, DIR)

# download the data
DF = 0  # download Flag
if DF:
    # Create a range of dates
    FHDATES = pd.date_range(
        start="2019-12-31 00:00",
        end="2021-01-02 01:00",
        freq="1H",
    )

    # Create a range of forecast lead times
    fxx = [0, 1]
    FH = FastHerbie(
        FHDATES,
        model="hrrr",
        fxx=fxx,
        product="subh",
        save_dir=DIR,
    )

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
wind_farms = pd.read_excel("In_windsolarlocations.xlsx")
wind_farms.loc[
    :, ["Predominant Turbine Model Number", "Predominant Turbine Manufacturer"]
] = "q"
wind_farms.loc[:, "Turbine Hub Height (Feet)"] = 328


# solar
solar_plantx = pd.read_excel("solar_plants.xlsx")
# add state name to lat long of the solar_plant
def city_state_country(row):
    coord = f"{row['lat']}, {row['lon']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw["address"]
    city = address.get("city", "")
    state = address.get("state", "")
    country = address.get("country", "")
    row["state"] = state
    # row['city'] = city
    # row['country'] = country
    return row


df = solar_plantx.apply(city_state_country, axis=1)
solar_plant = pd.merge(
    df,
    pd.DataFrame({"state": abv2state.values(), "state_abv": abv2state.keys()}),
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

# calculate output power
wind_power = {}
wind_speed_data = {}
solar_tmp_data = {}
solar_rad_data = {}
solar_vbd_data = {}
solar_vdd_data = {}

WindRead = 0
if WindRead:
    for inx, DEFAULT_HOURS_FORECASTED in enumerate(["0", "1"]):
        # for inx, DEFAULT_HOURS_FORECASTED in enumerate(["1"]):
        print("*" * 180)
        print(f"for wind power {DEFAULT_HOURS_FORECASTED}")
        print("*" * 180)
        wind_power[inx] = calculate_pout_individual(
            wind_farms,
            start_dt=START,
            end_dt=END,
            directory=DIR,
            hours_forecasted=DEFAULT_HOURS_FORECASTED,
        )

    # merge the output files
    wind_power[1].loc[wind_power[1].index.minute == 0, :] = wind_power[0].loc[
        wind_power[0].index.minute == 0, :
    ]
    wind_power[1].to_excel("wind_power.xlsx")

SolarRead = 0
if SolarRead:
    print("for solar power")
    for inx, DEFAULT_HOURS_FORECASTED in enumerate(["0", "1"]):
        a, b, c, d, e = extract_solar_data(
            # solar_plant.iloc[[0],:],
            solar_plant,
            start_dt=START,
            end_dt=END,
            directory=DIR,
            hours_forecasted=DEFAULT_HOURS_FORECASTED,
        )
        wind_speed_data[inx] = a
        solar_tmp_data[inx] = b
        solar_rad_data[inx] = c
        solar_vbd_data[inx] = d
        solar_vdd_data[inx] = e

    wind_speed_data[1].loc[wind_speed_data[1].index.minute == 0, :] = wind_speed_data[
        0
    ].loc[wind_speed_data[0].index.minute == 0, :]
    solar_tmp_data[1].loc[solar_tmp_data[1].index.minute == 0, :] = solar_tmp_data[
        0
    ].loc[solar_tmp_data[0].index.minute == 0, :]
    solar_rad_data[1].loc[solar_rad_data[1].index.minute == 0, :] = solar_rad_data[
        0
    ].loc[solar_rad_data[0].index.minute == 0, :]
    solar_vdd_data[1].loc[solar_vdd_data[1].index.minute == 0, :] = solar_vdd_data[
        0
    ].loc[solar_vdd_data[0].index.minute == 0, :]
    solar_vbd_data[1].loc[solar_vbd_data[1].index.minute == 0, :] = solar_vbd_data[
        0
    ].loc[solar_vbd_data[0].index.minute == 0, :]

    wind_speed_data[1].to_excel("solar_wind_speed.xlsx")
    solar_tmp_data[1].to_excel("solar_tmp_data.xlsx")
    solar_rad_data[1].to_excel("solar_rad_data.xlsx")
    solar_vdd_data[1].to_excel("solar_vdd_data.xlsx")
    solar_vbd_data[1].to_excel("solar_vbd_data.xlsx")


wind_speed_data = pd.read_excel("solar_wind_speed.xlsx", index_col=0)
wind_speed_data[wind_speed_data < 0] = 0
solar_tmp_data = pd.read_excel("solar_tmp_data.xlsx", index_col=0)
solar_tmp_data[solar_tmp_data < 0] = 0
solar_rad_data = pd.read_excel("solar_rad_data.xlsx", index_col=0)
solar_rad_data[solar_rad_data < 0] = 0
solar_vdd_data = pd.read_excel("solar_vdd_data.xlsx", index_col=0)
solar_vdd_data[solar_vdd_data < 0] = 0
solar_vbd_data = pd.read_excel("solar_vbd_data.xlsx", index_col=0)
solar_vbd_data[solar_vbd_data < 0] = 0

solar_data_all = {
    "wspd": wind_speed_data,
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

    # shift 6 hours UTC to CST
    dfs = dfs.shift(periods=-6)
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
    dfsd["tz"] = -6
    dfsd["elev"] = 898

    dfsa[inx] = dfsd
    if dfs.isnull().values.any():
        print(f"{lat} and {lon} at {inx} has NaN")


def calculate_power(solar_data, pv_dict):
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

# df = pd.read_csv("mysolardata.csv")
# lat = df["lat"][0]
# lon = df["lon"][0]
# tz = df["tz"][0]
# elev = df["elev"][0]
# df = df.drop(columns=["lat", "lon", "tz", "elev"])
# df.loc[df["dn"] > 1000, "dn"] = 1000
# df.loc[df["df"] > 1000, "df"] = 1000
# dfd = df.to_dict(orient="list")
# dfd["lat"] = lat
# dfd["lon"] = lon
# dfd["tz"] = tz
# dfd["elev"] = elev


dff = {}
for k, dfd in dfsa.items():
    print(f"calculate power for plant number {k}")
    power = calculate_power(dfd, pv_dict)
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


df.round(3).to_csv(
    "solaroutput.csv",
)

ipdb.set_trace()
