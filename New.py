
""" For this code to work you need to change the core.py of Herbie
The file names must be changed. search for localfilename
However, if we don't change this it seems that there are
two files that are being downloaded, why?
It is located in /local/alij/anaconda3/envs/herbie/lib/python3.11/site-packages/herbie
"""
import os

# I need to calculate the solar output here
import pandas as pd

import utils
from prereise.gather.const import (DATADIR, END, POINTSFN,  # OUTDIR,; YEAR,
                                   SEARCHSTRING, SELECTORS, START, TZ, YEAR,
                                   abv2state)

# utils.download_data(START, END, DATADIR, SEARCHSTRING)
points, wind_farms, solar_plants = utils.get_points(POINTSFN)
data = utils.read_data(points, START, END, DATADIR, SELECTORS)

# Replace with the actual path to your folder
folder_path = "./"
parquet_files = [file.split(".")[0] for file in os.listdir(
    folder_path) if file.endswith(".parquet")]

# reading the data locally from parquet files instead of grib files
data = {}
# read the data locally
for FN in parquet_files:
    df = pd.read_parquet(FN+".parquet")
    df.columns = df.columns.astype(int)
    data[FN] = df
print('read the data successfully')

# convert the wind speed dat to wind_farms
wind_output_power = utils.calculate_wind_pout(
    data['Wind80'], wind_farms, START, END, TZ).round(2)


# solar part
solar_output_power = utils.prepare_calculate_solar_power(
    data['Wind10'],
    data['2tmp'],
    data['vdd'],
    data['vbd'],
    solar_plants,
    YEAR,
)


__import__("ipdb").set_trace()
