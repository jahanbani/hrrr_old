# this reads a grib file to see what's inside
import pygrib
import ipdb

fn = '/home/alij/data/hrrr/20210101/subset_23ef0647__hrrr.t09z.wrfsubhf00.grib2'
gribs = pygrib.open(fn)

ipdb.set_trace()
