import pandas as pd
# Map state abbreviations to state name
abv2state = {
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}

# wind for wind farms
SELECTORS = {
    "UWind80": "U component of wind",
    "VWind80": "V component of wind",
    # "UWind10": "10 metre U wind component",
    # "VWind10": "10 metre V wind component",
    # "rad": "Downward short-wave radiation flux",
    # "vbd": "Visible Beam Downward Solar Flux",
    # "vdd": "Visible Diffuse Downward Solar Flux",
    # "2tmp": "2 metre temperature",
}

# OUTDIR = "../psse/grg-pssedata/"
OUTDIR = "./"

# study year; which year to study
YEAR = 2020
DIR = r"/research/alij/"
START = pd.to_datetime("2019-12-30 00:00")
END = pd.to_datetime("2021-01-05 01:00")
POINTSFN = "../psse/InputData/In_dut_80m_Iowa_Wind_Turbines.csv"
SEARCHSTRING = "V[B,D]DSF|DSWRF|TMP:2 m|(?:U|V)GRD:(?:10|80) m"
