# 3. TODO XXX look at the linear funciton; it is used for wind speed why not for other solar params
""" For this code to work you need to change the core.py of Herbie
The file names must be changed. search for localfilename
However, if we don't change this it seems that there are
two files that are being downloaded, why?
It is located in /local/alij/anaconda3/envs/herbie/lib/python3.11/site-packages/herbie
"""
# I need to calculate the solar output here
from IPython import embed
from powersimdata.input.grid import Grid
from prereise.gather.const import (
    abv2state,
    SELECTORS,
    DATADIR,
    OUTDIR,
    START,
    END,
    YEAR,
    SEARCHSTRING,
    POINTSFN,
)
from prereise.gather.solardata.helpers import to_reise
from prereise.gather.solardata.nsrdb import naive
# from prereise.gather.solardata.nsrdb.sam import (
#     retrieve_data_blended,
#     retrieve_data_individual,
#     retrieve_data_individual_ali,
#     retrieve_data_individual_orig,
# )
from prereise.gather.winddata.hrrr.hrrr import retrieve_data
from tqdm import tqdm
import utils


points, wind_farms, solar_plants = utils.get_points(POINTSFN)
data = utils.read_data(points, START, END, DATADIR, SELECTORS)

__import__("ipdb").set_trace()
