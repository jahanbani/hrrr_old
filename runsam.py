import numpy as np
import pandas as pd
import PySAM.Pvwattsv8 as PVWatts
import PySAM.ResourceTools as RT
import PySAM.PySSC as pssc  # noqa: N813
from tqdm import tqdm
import ipdb


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

df = pd.read_csv("mysolardata.csv")
lat = df["lat"][0]
lon = df["lon"][0]
tz = df["tz"][0]
elev = df["elev"][0]
df = df.drop(columns=["lat", "lon", "tz", "elev"])

df.loc[df["dn"] > 1000, "dn"] = 1000
df.loc[df["df"] > 1000, "df"] = 1000

dfd = df.to_dict(orient="list")
dfd["lat"] = lat
dfd["lon"] = lon
dfd["tz"] = tz
dfd["elev"] = elev



power = calculate_power(dfd, pv_dict)
dff = pd.DataFrame(power)
print(dff.loc[dff[0] > 0])

ipdb.set_trace()
