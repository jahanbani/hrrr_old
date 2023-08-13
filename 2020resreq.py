import os
from datetime import datetime

import gams
import numpy as np
import pandas as pd

import utils

YEAR = 2020
# set up gams working directory
HOME = os.path.expanduser("~")
gamsdir = r"/usr/local/gams/gams38.2_linux_x64_64_sfx"
ws = gams.GamsWorkspace(
    working_directory=r"/home/alij/hrrr",
    system_directory=gamsdir,
    debug=2,
)
opt = ws.add_options()
opt.optfile = 1
# gdxfile = r"Ali_p.gdx"
gdxfile = r"data.gdx"
datafile = "datanew.xlsx"
tt = {}
reservesdict = {}

start_dt = datetime.fromisoformat(str(YEAR) + "-01-01 00:00")
end_dt = datetime.fromisoformat(str(YEAR + 1) + "-01-01 00:00")


bustech = pd.read_excel(datafile, sheet_name=["Buses", "Tech"])
busgeo = bustech["Buses"]
bus_id_map = busgeo["Bus"].to_dict()
techs = bustech["Tech"][["Tech", "IsWind", "IsSolar"]]

dfs = utils.read_gdx(ws, gdxfile)


dfvarsorg = utils.vars_only(dfs, varsonly=False)
GAMSCOLS, selvars = utils.get_colvars()
dfvars = utils.rename_cols(dfvarsorg, selvars, GAMSCOLS)

dfvars["Gens"].columns = ["Bus", "Tech", "Gid", "Char", "level"]
gencap = dfvars["Gens"]
gencap["Bus"] = gencap["Bus"].astype(int)

gencap = gencap.loc[gencap["Char"] == "Cap", :].drop(columns=["Char"])

gens = pd.merge(pd.merge(gencap, techs, on="Tech"), busgeo, on="Bus")

gensws = gens.loc[((gens["IsWind"] == 1) | (gens["IsSolar"] == 1))].reset_index(
    drop=True
)
wind_farms_year = gensws.loc[gensws["IsWind"] == 1]
solar_plants_year = gensws.loc[gensws["IsSolar"] == 1]


# ###################Load data fetching########################################

print("doing load")
# get load data from ferc 714 for MISO for year 2020
ldf = utils.get_load(sdt=start_dt, edt=end_dt, area=321, directory=r"./load/")

# convert to 15 minutes
ldf = ldf.reindex(
    pd.date_range(start=start_dt, end=end_dt, freq="15T", inclusive="both")
)
ldf = ldf.interpolate()

ldf = ldf[ldf.index.year == YEAR]


# for solar level multiplication
solar_bus_each_year = (
    solar_plants_year[["Bus", "level"]].set_index(
        "Bus").groupby("Bus").sum("level")
)
solar_bus_each_year = solar_bus_each_year.loc[solar_bus_each_year.index > 200]

wind_bus_each_year = (
    wind_farms_year[["Bus", "level"]].set_index(
        "Bus").groupby("Bus").sum("level")
)

wind_bus_each_year = wind_bus_each_year.loc[wind_bus_each_year.index > 200]

dfow = pd.read_excel("wind_power.xlsx", index_col=0)
dfow.index = pd.to_datetime(dfow.index)
dfow = dfow.loc[dfow.index.year == YEAR, :]
dfow.columns = dfow.columns.astype(int)

dfow.rename(columns=bus_id_map, inplace=True)
# XXX this needs to be moved to windoutput.py but for now I am doing it here
# we are averaging the cases where multiple buses are being mapped with lat/lon
# I am not clear exactly what's happening here XXX
# https://stackoverflow.com/questions/40311987/pandas-mean-of-columns-with-the-same-names
df2 = dfow.transpose()
df2 = df2.groupby(by=df2.index, axis=0).apply(lambda g: g.mean())
dfw = df2.transpose()

wind_farms_year_prod = (
    dfw[wind_bus_each_year.index].mul(wind_bus_each_year["level"]).sum(axis=1)
)

# ################################Solar########################################
print("doing solar")

dfos = pd.read_csv("solaroutput.csv", index_col=0)
dfos.index = pd.to_datetime(dfos.index)
dfos = dfos.loc[dfos.index.year == YEAR, :]
dfos.columns = dfos.columns.astype(int)
dfos.rename(columns=bus_id_map, inplace=True)
# XXX this needs to be moved to solaroutput.py but for now I am doing it here
# we are averaging the cases where multiple buses are being mapped with lat/lon
# I am not clear exactly what's happening here XXX
# https://stackoverflow.com/questions/40311987/pandas-mean-of-columns-with-the-same-names
df2 = dfos.transpose()
df2 = df2.groupby(by=df2.index, axis=0).apply(lambda g: g.mean())
dfs = df2.transpose()

# each includes the generation of that year; 2026, 2032, 2038
solar_farms_year_prod = (
    dfs[solar_bus_each_year.index].mul(
        solar_bus_each_year["level"]).sum(axis=1)
)

nldiff = {}

netload = ldf-wind_farms_year_prod - solar_farms_year_prod

nldiff['15'] = netload.diff()

netload30 = netload.resample("30T").agg("first")

nldiff['30'] = netload30.diff()

std15 = np.std(nldiff['15'])
std30 = np.std(nldiff['30'])

reg = 3 * std15 * 1.65 / 3
ramp = 3 * std15 * 2 / 3 * 1.2
strr = 3 * std30

