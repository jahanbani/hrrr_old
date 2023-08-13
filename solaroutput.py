# I need to calculate the solar output here
import ipdb
from datetime import datetime

from prereise.gather.solardata.nsrdb.sam import (
    retrieve_data_individual,
    retrieve_data_blended,
)
from powersimdata.input.grid import Grid
from prereise.gather.solardata.nsrdb import naive
from prereise.gather.solardata.helpers import to_reise
from prereise.gather.const import abv2state
from prereise.gather.solardata.nsrdb.sam import retrieve_data_individual

import pandas as pd
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapiExercises")


def create_mock_pv_info():
    """Creates PV info data frame.

    :return: (*pandas.DataFrame*) -- mock PV info.
    """
    plant_code = [1, 2, 3, 4, 5]
    state = ["UT", "WA", "CA", "CA", "CA"]
    capacity = [10, 5, 1, 2, 3]
    single = ["N", "Y", "Y", "Y", "Y"]
    dual = ["Y", "N", "N", "N", "Y"]
    fix = ["N", "Y", "Y", "N", "N"]

    pv_info = pd.DataFrame(
        {
            "State": state,
            "Nameplate Capacity (MW)": capacity,
            "Single-Axis Tracking?": single,
            "Dual-Axis Tracking?": dual,
            "Fixed Tilt?": fix,
            "Plant Code": plant_code,
        }
    )

    return pv_info


def download_wind_solar():
    # check if it is solar or wind, check the api file
    from prereise.gather.winddata.hrrr.hrrr import retrieve_data

    print("download started")
    # download all the data
    retrieve_data(start_dt=start_dt, end_dt=end_dt, directory=directory)

    print("download finished")
    return


# needed to download from NSRDB only
email = "ali_jahanbani@yahoo.com"
api_key = "mx63fHiE5o4c6amHcdmzfe22t0NTlNDgsVxdreea"

if True:
    grid = Grid("Texas")
    solar_plant = grid.plant.groupby("type").get_group("solar")
    solar_plant["state_abv"] = "TX"

    df = retrieve_data_blended(
        email,
        api_key,
        grid=None,
        solar_plant=solar_plant,
        interconnect_to_state_abvs=abv2state,
        year="2016",
        rate_limit=0.5,
        cache_dir=None,
    )

    ipdb.set_trace()

# start and end date
start_dt = datetime.fromisoformat("2019-01-01")
end_dt = datetime.fromisoformat("2020-01-02")

# where to read the solar data
directory = "./data/"

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

df = retrieve_data_blended(
    email,
    api_key,
    grid=None,
    solar_plant=solar_plant,
    interconnect_to_state_abvs=abv2state,
    year="2016",
    rate_limit=0.5,
    cache_dir=None,
)

ipdb.set_trace()
