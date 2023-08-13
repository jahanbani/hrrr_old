# two things we do here
# 1. download the data
# 2. calculate output power of wind/solar
# 3. XXX look at the linear funciton; it is used for wind speed why not for other solar params


from datetime import datetime

# I need to calculate the solar output here
import ipdb
import pandas as pd
from geopy.geocoders import Nominatim
from herbie import Herbie
from herbie.tools import FastHerbie
from IPython import embed
from powersimdata.input.grid import Grid

from prereise.gather.const import abv2state
from prereise.gather.solardata.helpers import to_reise
from prereise.gather.solardata.nsrdb import naive
from prereise.gather.solardata.nsrdb.sam import (
    retrieve_data_individual_ali,
    retrieve_data_individual,
)
from prereise.gather.winddata.hrrr.calculations import (
    calculate_pout_individual,
    extract_solar_data,
)
from prereise.gather.winddata.hrrr.hrrr import retrieve_data

geolocator = Nominatim(user_agent="geoapiExercises")

DIR = r"/research/alij/Solar"
START = pd.to_datetime("2021-01-01 00:00")
END = pd.to_datetime("2022-01-01 01:00")

# retrieve_data(START, END, DIR)

# download the data
DF = 0  # download Flag
if DF:
    # Create a range of dates
    FHDATES = pd.date_range(
        start="2021-01-01 00:00",
        end="2022-01-01 01:00",
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
        # searchString="V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m",
        # wind only
        searchString="(?:U|V)GRD:80 m",
        # solar only
        # searchString="V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:10 m",

        save_dir=DIR,
        max_threads=200,
        verbose=True,
    )
    import ipdb; ipdb.set_trace()


# wind
wind_farms = pd.read_excel("wind_farms.xlsx")
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
solar_plant.index.name = 'plant_id'

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
        print('*' * 180)
        print(f'for wind power {DEFAULT_HOURS_FORECASTED}')
        print('*' * 180)
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

    ipdb.set_trace()

print('for solar power')
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
solar_tmp_data[1].loc[solar_tmp_data[1].index.minute == 0, :] = solar_tmp_data[0].loc[
    solar_tmp_data[0].index.minute == 0, :
]
solar_rad_data[1].loc[solar_rad_data[1].index.minute == 0, :] = solar_rad_data[0].loc[
    solar_rad_data[0].index.minute == 0, :
]
solar_vdd_data[1].loc[solar_vdd_data[1].index.minute == 0, :] = solar_vdd_data[0].loc[
    solar_vdd_data[0].index.minute == 0, :
]
solar_vbd_data[1].loc[solar_vbd_data[1].index.minute == 0, :] = solar_vbd_data[0].loc[
    solar_vbd_data[0].index.minute == 0, :
]

wind_speed_data[1].to_excel("solar_wind_speed.xlsx")
solar_tmp_data[1].to_excel("solar_tmp_data.xlsx")
solar_rad_data[1].to_excel("solar_rad_data.xlsx")
solar_vdd_data[1].to_excel("solar_vdd_data.xlsx")
solar_vbd_data[1].to_excel("solar_vbd_data.xlsx")


ipdb.set_trace()


wind_speed_data = pd.read_excel("solar_wind_speed.xlsx", index_col=0)
solar_tmp_data = pd.read_excel("solar_tmp_data.xlsx", index_col=0)
solar_rad_data = pd.read_excel("solar_rad_data.xlsx", index_col=0)
solar_vdd_data = pd.read_excel("solar_vdd_data.xlsx", index_col=0)
solar_vbd_data = pd.read_excel("solar_vbd_data.xlsx", index_col=0)


solar_data_all = {
    "wspd": wind_speed_data,
    "df": solar_vdd_data,
    "dn": solar_vbd_data,
    "tdry": solar_tmp_data - 273.15,
}

latlon = solar_plant.loc[:, ["lat", "lon"]]


def prepare_solar_data(dfd, latlon):
    """prepare input for pvwatts
    :input: dictionary of wind speed, dhi, dni, temp
    each are datetime index and plant columns
    :return: dictionary of each plant, datetime index, wind speed, dhi, dni, temp columns

    :df.index.name='plant_id'
    columns required: lat, lon, Fixed Tilt? Single-Axis Tracking? Dual-Axis Tracking?
    """
    lat = latlon["lat"]
    lon = latlon["lon"]
    df = pd.concat(dfd.values(), keys=dfd.keys())

    dfsa = {}
    for inx, (lat, lon) in enumerate(latlon.values):
        print(f"for index {inx}")
        dfs = df[[inx]].unstack(0)[inx]
        dfs.loc[:, "lat"] = lat
        dfs.loc[:, "lon"] = lon
        dfs.loc[:, "tz"] = -6
        dfs.loc[:, "elev"] = 898
        dfs.loc[:, "year"] = dfs.index.year
        dfs.loc[:, "month"] = dfs.index.month
        dfs.loc[:, "day"] = dfs.index.day
        dfs.loc[:, "hour"] = dfs.index.hour
        dfs.loc[:, "minute"] = dfs.index.minute
        dfsa[inx] = dfs.reset_index(drop=True)
    return dfsa


solar_data = prepare_solar_data(solar_data_all, latlon)


# needed to download from NSRDB only
email = "ali_jahanbani@yahoo.com"
api_key = "mx63fHiE5o4c6amHcdmzfe22t0NTlNDgsVxdreea"

if True:
    # grid = Grid("Texas")
    # solar_plant = grid.plant.groupby("type").get_group("solar")
    # solar_plant["state_abv"] = "TX"

    # solar_plant.loc[:, "Fixed Tilt?"] = True
    # solar_plant.loc[:, ["Single-Axis Tracking?", "Dual-Axis Tracking?"]] = False

    df = retrieve_data_individual(
        email,
        api_key,
        # grid=None,
        solar_data,
        solar_plant=solar_plant,
        # interconnect_to_state_abvs=abv2state,
        year="2021",
        rate_limit=0.5,
        cache_dir=None,
    )

    ipdb.set_trace()
# calculate output power
solar_power = {}
solar_power = retrieve_data_individual(
    email, api_key, solar_plant, year="2020", rate_limit=0.5, cache_dir=None
)

ipdb.set_trace()
