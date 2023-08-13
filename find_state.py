from geopy.geocoders import Bing
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import concurrent.futures
import utils

def get_location_by_coordinates(lat, lon):
    location = geocode((lat, lon))
    return location.raw['address'].get('adminDistrict', '')


def process_df(df):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        states = list(executor.map(
            get_location_by_coordinates, df['lat'], df['lon']))
    df['state'] = states
    return df


# read the In_windsolarlocations.xlsx
df = pd.read_excel('In_windsolarlocations.xlsx')[['lat', 'lon']]
df = process_df(df)
print(df)
__import__('ipdb').set_trace()
