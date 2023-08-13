from herbie.tools import FastHerbie
from herbie import Herbie

import pandas as pd
import ipdb

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
    save_dir="/local/alij/data",
)

FH.download(
    searchString="V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m",
    save_dir="/local/alij/data",
    max_threads=200,
    verbose=True,
)

ipdb.set_trace()

# H = Herbie(
#     "2020-01-01 00:00",
#     model="hrrr",  # model name
#     product="subh",  # model produce name (model dependent)
#     fxx=0,  # forecast lead time
# )

# ds = H.xarray(
#     searchString="V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m", remove_grib=False
# )
