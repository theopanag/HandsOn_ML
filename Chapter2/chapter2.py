from pathlib import Path
import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import tarfile
from urllib import request

import pdb

####### LOAD THE DATA AND EXPLORE ########

# Download the data from github
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Directory to store/extract data
DATA_DIR = Path(__file__).parent.joinpath('Data')
FIGURE_DIR = Path(__file__).parent.joinpath('Figures')


# Function to separate functionalitites in script

def print_section(section_name):
    print("#" * (22 + len(section_name)))
    print_str = " ".join(["#" * 10, section_name, "#" * 10])
    print(print_str)
    print("#" * (22 + len(section_name)))


# Function to fetch data

def fetch_housing_data(
        housing_url=HOUSING_URL,
        housing_path=DATA_DIR
):
    housing_path.mkdir(parents=True, exist_ok=True)
    tgz_path = housing_path.joinpath("housing.tgz")
    request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


# Function to load the data to memory

def load_housing_data(housing_path=DATA_DIR):
    return pd.read_csv(housing_path.joinpath("housing.csv"))


# Run the main function
if __name__ == "__main__":

    # Initialize the figures' directory if it doesn't exist
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch the data from the web and save extracted file
    fetch_housing_data()

    # Load the saved data
    df = load_housing_data()

    print_section("DATA EXPLORATION")
    print()
    pprint.pprint(df.head())
    print()
    pprint.pprint(df.info())
    print()
    pprint.pprint(df.describe())
    print()

    fig = plt.figure()
    df.hist(bins=50, figsize=(20, 15))
    plt.savefig(FIGURE_DIR.joinpath("histogram_of_original_data.png"))

    print_section("TRAIN-TEST SPLIT")

    # Create the 'income_cat' attribute to base stratified split upon
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0., 1.5, 3., 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    fig = plt.figure()
    df.income_cat.hist()
    plt.savefig(FIGURE_DIR.joinpath("histogram_of_income_categories.png"))

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    # Remove temporary 'income_cat' attribute after performing stratified split
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    print_section("VISUALIZE DATA TO GAIN INSIGHTS")

    # Create a copy of the training set to play with, without harming the actual data
    housing = strat_train_set.copy()

    # Visualize geographical data
    fig = plt.figure()
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.1
    )
    plt.savefig(FIGURE_DIR.joinpath("geographical_scatterplot.png"))

    fig = plt.figure()
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        alpha=0.4,
        s=housing["population"] / 100,
        label="population",
        figsize=(10, 7),
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=True
    )
    plt.savefig(FIGURE_DIR.joinpath("locationVSpopulation.png"))

    # Explore correlations with 'median house value'

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    pprint.pprint(housing.corr()["median_house_value"].sort_values(ascending=False))

    print_section("PREPARE DATA FOR ML")

    # Separate features and labels in the training set to avoid applying transformations to target values
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    # Only numerical columns can be used for imputation
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)

    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(
        X,
        columns=housing_num.columns,
        index=housing_num.index
    )

    pprint.pprint(housing_tr.head())

    # One-Hot encode the nominal features ('ocean proximity' in our case)
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing[["ocean_proximity"]])

    print(cat_encoder.categories_)