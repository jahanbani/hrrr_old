import os
import random
import shutil
import sys
import time
from datetime import datetime

import folium
import gams
import gamstransfer as gt
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from IPython import embed
from powersimdata.input.grid import Grid
from pylab import MaxNLocator
from tqdm import tqdm

import utils
from prereise.gather.winddata.hrrr.calculations import calculate_pout_individual
from prereise.gather.winddata.hrrr.constants import (
    DEFAULT_HOURS_FORECASTED,
    DEFAULT_PRODUCT,
)

if os.path.isfile("maindf.xlsx"):
    os.remove("maindf.xlsx")


# directory = "./data/"
figdir = "./figs/"


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
gdxfile = r"Ali_p.gdx"
datafile = "datanew.xlsx"

gdxfiledata = "data.gdx"
# os.rename(gdxfiledata, gdxfiledata + "_orig")


tt = {}
reservesdict = {}
for i in range(0, 10):
    print(f"we are in iteration {i+1} out of 10")
    tt0 = time.time()
    t1 = ws.add_job_from_file(r"/home/alij/hrrr/acep.gms")
    t1.run(opt)
    tt[("CEP", i)] = time.time()
    print("CEP done in {0} seconds.".format(time.time() - tt0))
    # if True:
    read_from_gdx = 1
    if read_from_gdx == 1:
        bustech = pd.read_excel(datafile, sheet_name=["Buses", "Tech"])
        busgeo = bustech["Buses"]
        bus_id_map = busgeo["Bus"].to_dict()
        techs = bustech["Tech"][["Tech", "IsWind", "IsSolar"]]

        dfs = utils.read_gdx(ws, gdxfile)
        # make a copy of the gdx file so we have it for later
        os.rename(gdxfile, str(i) + gdxfile)

        dfvarsorg = utils.vars_only(dfs)

        GAMSCOLS, selvars = utils.get_colvars()

        dfvars = utils.rename_cols(dfvarsorg, selvars, GAMSCOLS)
        gencap = dfvars["pv_GenCapTot"][["Bus", "Tech", "Year", "level"]]
        gens = pd.merge(pd.merge(gencap, techs, on="Tech"), busgeo, on="Bus")

        wind_farms_year = {}
        solar_plants_year = {}
        for year in [2020, 2026, 2032, 2038]:
            gensws = gens.loc[
                ((gens["IsWind"] == 1) | (gens["IsSolar"] == 1))
                & (gens["Year"] == str(year))
            ].reset_index(drop=True)
            wind_farms_year[year] = gensws.loc[gensws["IsWind"] == 1]
            solar_plants_year[year] = gensws.loc[gensws["IsSolar"] == 1]

        for k, v in wind_farms_year.items():
            utils.append_df_to_excel(
                "wind_farms.xlsx",
                v,
                sheet_name="Year" + str(k) + "_Iter" + str(i),
                index=False,
            )
        for k, v in solar_plants_year.items():
            utils.append_df_to_excel(
                "solar_plants_year.xlsx",
                v,
                sheet_name="Year" + str(k) + "_Iter" + str(i),
                index=False,
            )
    else:
        wind_farms_year = pd.read_excel("wind_farms.xlsx", sheet_name=None)
        solar_plants_year = pd.read_excel("solar_plants_year.xlsx", sheet_name=None)

    # for wind level multiplication
    wind_bus_each_year = {
        k: wind_farms_year[k][["Bus", "level"]]
        .set_index("Bus")
        .groupby("Bus")
        .sum("level")
        for k, v in wind_farms_year.items()
    }

    # for solar level multiplication
    solar_bus_each_year = {
        k: solar_plants_year[k][["Bus", "level"]]
        .set_index("Bus")
        .groupby("Bus")
        .sum("level")
        for k, v in solar_plants_year.items()
    }

    wind_farmsall = pd.concat(
        [wind_farms_year[k] for k, v in wind_farms_year.items()], ignore_index=True
    )
    solar_farmsall = pd.concat(
        [solar_plants_year[k] for k, v in solar_plants_year.items()], ignore_index=True
    )
    # for renaming
    wind_bus_id_map = wind_farmsall["Bus"].to_dict()
    solar_bus_id_map = solar_farmsall["Bus"].to_dict()

    print("doing wind")

    wind_farms = wind_farmsall.drop_duplicates(subset=["Bus"])
    solar_farms = solar_farmsall.drop_duplicates(subset=["Bus"])

    start_dt = datetime.fromisoformat(str(YEAR) + "-01-01 00:00")
    end_dt = datetime.fromisoformat(str(YEAR + 1) + "-01-01 00:00")

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

    # each includes the generation of that year; 2026, 2032, 2038
    wind_farms_year_prod = {
        int(k): dfw[v.index].mul(wind_bus_each_year[k]["level"]).sum(axis=1)
        for k, v in wind_bus_each_year.items()
    }

    # ###################Load data fetching########################################

    print("doing load")
    # get load data from ferc 714 for MISO for year 2020
    ldf = utils.get_load(sdt=start_dt, edt=end_dt, area=321, directory=r"./load/")

    # convert to 15 minutes
    ldf = ldf.reindex(
        pd.date_range(start=start_dt, end=end_dt, freq="15T", inclusive="both")
    )
    ldf = ldf.interpolate()

    # apply the load growth XXX this needs to be revisited;
    # XXX  we accounted for pandemic effect in the input of the CEP
    LG = 1.067
    loaddf = {}
    for inx, year in enumerate([2020, 2026, 2032, 2038]):
        loaddf[year] = ldf * np.power(LG, inx)

    # ###################We Miss You Solar########################################
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
    solar_farms_year_prod = {
        int(k): dfs[v.index].mul(solar_bus_each_year[k]["level"]).sum(axis=1)
        for k, v in solar_bus_each_year.items()
    }

    # ###################calculate net load################
    print("calculate net load")
    maindf = {k: pd.DataFrame() for k, v in wind_farms_year_prod.items()}
    std = {}
    reserves = {}
    strdf = {}
    for k, v in wind_farms_year_prod.items():
        maindf[k]["W"] = v
        maindf[k]["S"] = solar_farms_year_prod[k]
        maindf[k]["L"] = loaddf[k]
        maindf[k]["NL"] = maindf[k]["L"] - maindf[k]["W"] - maindf[k]["S"]

        maindf[k].index = pd.to_datetime(maindf[k].index)
        strdf[k] = maindf[k].resample("30T").agg("first")
        strdf[k]["NL30"] = strdf[k]["L"] - strdf[k]["W"] - strdf[k]["S"]

        strdf[k]["WDiff30"] = strdf[k]["W"].diff()
        strdf[k]["SDiff30"] = strdf[k]["S"].diff()
        strdf[k]["LDiff30"] = strdf[k]["L"].diff()
        strdf[k]["NLDiff30"] = strdf[k]["NL30"].diff()

        maindf[k]["NLDiff"] = maindf[k]["NL"].diff()
        maindf[k]["WDiff"] = maindf[k]["W"].diff()
        maindf[k]["SDiff"] = maindf[k]["S"].diff()
        maindf[k]["LDiff"] = maindf[k]["L"].diff()

        std30 = np.std(strdf[k]["NLDiff30"])
        std15 = np.std(maindf[k]["NLDiff"])

        # std[k, "RegUp"] = [3 * std15 / 3 * 1.65]
        # std[k, "RegDwn"] = [3 * std15 / 3 * 1.65]
        # std[k, "RampUp"] = [3 * std15 * 2 / 3 * 1.2]
        # std[k, "RampDwn"] = [3 * std15 * 2 / 3 * 1.2]

        std[k, "RegUp"] = [3 * std15 / 3 * 0.6655]
        std[k, "RegDwn"] = [3 * std15 / 3 * 0.6655]
        std[k, "RampUp"] = [3 * std15 * 2 / 3 * 1.4966]
        std[k, "RampDwn"] = [3 * std15 * 2 / 3 * 1.4966]

        std[k, "STRUp"] = [3 * std30]
        std[k, "STRDwn"] = [3 * std30]

        # contingency should be read from previous year since it doesn't change;
        # reading it from this load is wrong; we changed the peak load in CEP data
        # to account for pandemic effect
        # std[k, "Cont"] = [3* loaddf[k].max() * 0.025]

    print(f"write maindf for year {k}")
    utils.append_df_to_excel_dfdict("maindf.xlsx", maindf, Iter=i, index=False)
    utils.append_df_to_excel_dfdict("strdf.xlsx", strdf, Iter=i, index=False)

    print("doing figures")
    ttbwp = time.time()
    for k, v in maindf.items():
        # 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        maindf[k]["NLDiff"].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NLDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["NLDiff"] / 3 * 1.65).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NLDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["NLDiff"] / 3 * 2 * 1.2).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NLDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (strdf[k]["NLDiff30"]).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NLDiff{k}_Itr{i}_30min.pdf")
        plt.close()

        # 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        maindf[k]["WDiff"].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"WindDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["WDiff"] / 3 * 1.65).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"WindDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["WDiff"] / 3 * 2 * 1.2).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"WindDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (strdf[k]["WDiff30"]).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"WindDiff{k}_Itr{i}_30min.pdf")
        plt.close()

        # solar 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        maindf[k]["SDiff"].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"SolarDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["SDiff"] / 3 * 1.65).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"SolarDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["SDiff"] / 3 * 2 * 1.2).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"SolarDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (strdf[k]["SDiff30"]).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"SolarDiff{k}_Itr{i}_30min.pdf")
        plt.close()

        # 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        maindf[k]["LDiff"].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"LoadDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["LDiff"] / 3 * 1.65).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"LoadDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (maindf[k]["LDiff"] / 3 * 2 * 1.2).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"LoadDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        (strdf[k]["LDiff30"]).hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"LoadDiff{k}_Itr{i}_30min.pdf")
        plt.close()

        # ############ NON ZEROS ##########################

        # WindVarNZ = WindVar[WindVar != 0]
        # 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["NLDiff"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZNLDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["NLDiff"] / 3 * 1.65
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZNLDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["NLDiff"] / 3 * 2 * 1.2
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZNLDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = strdf[k]["NLDiff30"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZNLDiff{k}_Itr{i}_30min.pdf")
        plt.close()

        # wind 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["WDiff"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZWindDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["WDiff"] / 3 * 1.65
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZWindDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["WDiff"] / 3 * 2 * 1.2
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZWindDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = strdf[k]["WDiff30"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZWindDiff{k}_Itr{i}_30min.pdf")
        plt.close()

        # solar 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["SDiff"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZSolarDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["SDiff"] / 3 * 1.65
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZSolarDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["SDiff"] / 3 * 2 * 1.2
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZSolarDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = strdf[k]["SDiff30"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZSolarDiff{k}_Itr{i}_30min.pdf")
        plt.close()

        # 15 minutes
        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["LDiff"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZLoadDiff{k}_Itr{i}_15min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["LDiff"] / 3 * 1.65
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZLoadDiff{k}_Itr{i}_5min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = maindf[k]["LDiff"] / 3 * 2 * 1.2
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZLoadDiff{k}_Itr{i}_10min.pdf")
        plt.close()

        fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
        ZZ = strdf[k]["LDiff30"]
        ZZ[ZZ != 0].hist(bins=100, ax=ax1)
        plt.savefig(figdir + f"NZLoadDiff{k}_Itr{i}_30min.pdf")
        plt.close()

    ttewp = time.time()
    tt[("Vis", i)] = ttewp - ttbwp
    print(f"writing and plotting took {ttewp-ttbwp} seconds.")

    # write the reserve requirement to the GDX file
    m = gt.Container(gdxfiledata, ws.system_directory)

    # we don't calculate the values for year 2020; we rather use what we had.
    # let us take that from data.gdx
    resex = m.data["ResReqStatic"].records
    resex.columns = ["ResType", "Year", "value"]
    resex["Year"] = resex["Year"].astype(int)
    res20 = resex.loc[(resex["Year"] == 2020) & (resex["ResType"] != "Cont")]
    rescont = resex.loc[resex["ResType"] == "Cont"].set_index(["Year", "ResType"])
    res20 = res20.set_index(["Year", "ResType"])
    res20dict = res20["value"].to_dict()
    rescontdict = rescont["value"].to_dict()

    # Python 3.9 and above
    # newstd = std | res20dict | rescontdict
    newstd = std | rescontdict

    reserves = pd.DataFrame(newstd).stack().droplevel(0).round(2)
    print("-" * 100)
    print(f"we are in iteration {i} and the reserves are like below")
    print(reserves)
    print("-" * 100)

    reservesdict[i] = reserves

    reserves.reset_index(drop=False, inplace=True)
    reserves = pd.melt(reserves, id_vars=["index"], value_vars=[2020, 2026, 2032, 2038])

    m.data["ResReqStatic"].setRecords(reserves)
    m.write(gdxfiledata)

    tte = time.time()
    print(f"iteration {i} done in {tte-tt0} seconds.")


if os.path.isfile("reserves.xlsx"):
    shutil.copy2("reserves.xlsx", "reserves_old.xlsx")
    os.remove("reserves.xlsx")

for k, v in reservesdict.items():
    utils.append_df_to_excel(
        "reserves.xlsx",
        v,
        sheet_name="Iter" + str(k),
        index=False,
    )

df = pd.concat(reservesdict.values(), keys=reservesdict.keys()).droplevel(1)
df.index.names = ["Iter"]
df.columns.names = ["Year"]
df = df.reset_index().set_index(["Iter", "index"])


# this is based on each year

dff = df.stack().unstack(2).reset_index()

for k, v in dff.groupby(["index"]):
    ndf = v.drop(columns=["index"])
    ax = ndf.plot(
        x="Iter",
        y=ndf.drop(columns=["Iter"]).columns,
        colormap="jet",
        marker=".",
        markersize=10,
        title=k,
    )
    ax1 = ax.get_xaxis()
    ax1.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(k + "year" + ".pdf")


# this is based on each reserve

dff = df.stack().unstack(1).reset_index()

for k, v in dff.groupby(["Iter"]):
    ndf = v.drop(columns=["Iter"])
    ax = ndf.plot(
        x="Year",
        y=ndf.drop(columns=["Year"]).columns,
        colormap="jet",
        marker=".",
        markersize=10,
        title=k,
    )
    ax1 = ax.get_xaxis()
    ax1.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(str(k) + "ResType" + ".pdf")


__import__('ipdb').set_trace()

# k=1.7 from 15 to 5;     sigma/3 * 1.65   ;;;;;;;;;;;;; 15 to 10   sigma * 2/3 * 1.2

# ######################################################################
# ########################Visualizaitons################################
# ######################################################################
# for k, v in wind_farms_year_prod.items():
#     # variability
#     WindVar = v.diff()
#     # WindVarPos = WindVar[WindVar >= 0]
#     # WindVarNeg = WindVar[WindVar < 0]
#     WindVarNZ = WindVar[WindVar != 0]
#     fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
#     fig.suptitle(f"Wind Variability in {k}")
#     WindVarNZ.hist(bins=50, ax=ax1)
#     plt.savefig(figdir + f"windvarsnz_{k}.pdf")
#     plt.close()

# ldf = loaddf[2038]
# # variability
# ldfvar = ldf.diff()
# ldfvarnz = ldfvar[ldfvar != 0]
# fig, ax1 = plt.subplots(ncols=1, figsize=(10, 5), constrained_layout=True)
# fig.suptitle("Load Changes")
# ldfvarnz.hist(bins=50, ax=ax1)
# plt.savefig(figdir + "loadvarsnz.pdf")
# plt.close()
