import os
import argparse

from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from neuroHarmonize import harmonizationLearn
from matplotlib import colormaps as cmaps
from smogn import smoter # SmoteR for regression-oriented oversampling

def abs_path(local_filename, data_folder):
    """
    Gets the absolute path of the file given the name of the folder containing the data
    and the name of the file inside that folder and
    assuming that the repository contains a data folder
    and a code folder.

    :param local_filename: name of the data file
    :type local_filename: str
    :param data_folder: name of the folder which contains the data
    :type data_folder: str
    :return: the function returns the absolute path of the selected file
    :rtype: str
    
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # path of the code

    # Construct the absolute path to the data directory relative to the code directory
    data_dir = os.path.join(script_dir, "..", data_folder)

    # Construct the absolute path to the data file
    data_file_path = os.path.join(data_dir, local_filename)

    return data_file_path

def mean_spurious(df):
    """
    Handles spurious 0 and -9999 values in the data, if the dataset is not properly cleaned before.
    
    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Cleaned DataFrame
    :rtype: pd.DataFrame
    """
    # Replace -9999 with NaN
    df.replace(-9999, np.nan, inplace=True)
    # Replace 0 with NaN
    df.replace(0, np.nan, inplace=True)
    # Fill NaN values with the mean of the respective columns
    df.fillna(df.mean(), inplace=True)
    return df


def check_for_spurious(df: pd.DataFrame, show: bool = False) -> pd.DataFrame:
    """
    Run this snippet to see what are columns with invalid values 0 or -9999
    
    :param df: Input dataframe containing extracted features
    :type df: pd.DataFrame
    :param show: Whether to show the dataset, default = False
    :type show: Bool

    Returns: Dataframe containing the names of columns with the most dirty data
    and counter of dirty data.

    """
    # Store a counter of instances of missing data for each feature
    dctSpurious = defaultdict(int)
    for i in range(df.shape[1]):
        for j in range(df.shape[0]):
            value = df.iloc[j, i]
            if value == -9999 or value == 0:
                dctSpurious[df.columns[i]] += 1

    # Create a pandas dataframe with the gathered data and
    dtfSpurious = pd.DataFrame([dctSpurious.keys(),
                                dctSpurious.values()])
    dtfSpurious = dtfSpurious.transpose()
    dtfSpurious.columns = ["Feature name", "Number of missing values"]
    if show:
        print(dtfSpurious)
    return dtfSpurious


def handle_spurious(df: pd.DataFrame, *args: str) -> pd.DataFrame:
    """
    Handles spurious 0 and -9999 values in the DataFrame by takeing the mean value of the column.
    
    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Cleaned DataFrame with spurious values handled
    :rtype: pd.DataFrame

    Args:
        *args (str): Names of columns to remove
    """

    for arg in args:
        if isinstance(arg, str):
            df.drop(arg, axis="columns", inplace=True)
        else:
            print("Invalid argument!")

    # Replace -9999 with NaN
    df.replace(-9999, np.nan, inplace=True)
    # Replace 0 with NaN
    df.replace(0, np.nan, inplace=True)

    # Fill NaN values with the mean of the respective columns
    # Temporarily disabled, first we have to make sure all entries are numbers
    # df.fillna(df.mean(), inplace=True)

    # Remove all other rows containing dirty data
    df.dropna(inplace=True)

    return df

def group_selection(array, group, value):
    indices = np.where(group == value)[0]
    selected = array[indices]
    return selected

def get_correlation(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Calculates the correlation between features using by default the "Pearson"
    method. Useful for reducing model complexity
    Args:
        df: Dataframe containing the features
        threshold: Sets the correlation threshold considered significant

    Returns: A dataframe containing the correlation coefficient
    of each feature against the others if above the threshold.

    """
    # Drop the first column. Only numerical values accepted
    correlation_dataframe = df.corr(numeric_only=True)

    # Look for high correlated features
    # List containing info on correlated features
    lstCorrelated = []
    # Iterate over the correlation matrix and check if there are correlation values over the user-set
    # threshold and add the respective features to a dataframe
    for i in range(correlation_dataframe.shape[0]):
        for column in correlation_dataframe.columns:
            value = correlation_dataframe[column].iat[i]
            if abs(value) > threshold:
                if value != 1:
                    # column is the name of the feature
                    # correlation_dataframe.index[i] is the feature comparing against
                    # value is the correlation coefficient between the above
                    lstCorrelated.append([column,
                                          correlation_dataframe.index[i],
                                          value]
                                         )
    # Create the dataframe
    dtfHighlyCorrelated = pd.DataFrame(lstCorrelated)
    dtfHighlyCorrelated.columns = ["Feature", "Against-feature", "corr-value"]

    return dtfHighlyCorrelated


def check_site_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if there are features influenced by the site where the image got shot.
    Checks whether the mean of each feature is similar across sites. If it's not, then there's bias in
    the image acquisition.

    Args:
        df: Pandas dataframe containing all the features labeled according the site of origin.

    Returns:
        A pandas dataframe containing the mean of each feature calculated separately for each site.

    """

    df_length = df.shape[0]
    # Iterate over the df and save in a list all site names
    lstSiteNames = []
    site_column = df.columns[0]
    for i in range(df_length):
        site_name = df[site_column].iat[i]
        site_name = site_name.split("_")[0]
        if site_name not in lstSiteNames:
            lstSiteNames.append(site_name)

    # Data structure for storing mean values
    dtfSiteFeatures = pd.DataFrame(np.zeros((len(df.columns[1:]), len(lstSiteNames))),
                                   index=df.columns[1:],
                                   columns=lstSiteNames)

    # Calculate the mean of each feature for each site separately.
    # Slow asf, needs refactoring.
    temp_feature_value = []
    for site in lstSiteNames:
        for feature in df.columns[1:]:
            for i in range(df_length):
                if site in df[site_column].iat[i]:
                    temp_feature_value.append(df[feature].iat[i])
            dtfSiteFeatures[site][feature] = np.mean(temp_feature_value)
            # Resets the array
            temp_feature_value = []

    return dtfSiteFeatures


def get_data(filename, target_col, ex_cols=0, **kwargs):
    """
    Obtains the features and target arrays from a CSV file. Optionally harmonizes the data 
    using neuroHarmonize and includes additional columns for grouping.

    :param filename: Path to the CSV file.
    :type filename: str
    :param target_col: Name of the target column.
    :type target_col: str
    :param ex_cols: Number of initial columns to exclude from the features (default is 0).
    :type ex_cols: int, optional
    :param kwargs: Additional keyword arguments:
                   - group_col: Name of the group column (optional).
                   - site_col: Name of the site column for harmonization (optional).
                   -overs: Boolean flag in order to perform SmoteR oversampling (optional).
    :return: NumPy arrays of features, targets, and optionally the group.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray or None)
    """
    group_col = kwargs.get('group_col', None)
    site_col = kwargs.get('site_col', None)
    overs = kwargs.get('overs', False)
    logger.info(f'Reading {os.path.basename(filename)} with {target_col} as target column')
    # Importing data from csv file as data
    data = pd.read_csv(filename, delimiter = ';')
    if overs:
        data = smoter(data, target_col)
        logger.info('Oversampling performed with SmoteR')
    # Excluding the first ex_cols columns
    features_df = data.iloc[:, ex_cols:]
    # Convert in to numpy matrix
    features_df = mean_spurious(features_df) # Removing (hopefully absent) spurious values
    features = features_df.values
    # Target array (numpy.ndarray)
    targets = data[target_col].values
    if site_col in data.columns:
        covars = data[[site_col]]
        covars.loc[:, site_col] = covars[site_col].str.rsplit('_', n=1).str[0]
        covars.rename(columns={site_col: 'SITE'}, inplace=True)  # Rename the column
        _, features = harmonizationLearn(features, covars)
        logger.info('Harmonizing data with neuroHarmonize ')

    if len(features) != len(targets):
        logger.error("Number of samples in features and targets do not match ")
        raise ValueError("Mismatch between number of features and targets samples")
    if group_col:
        logger.info(" Splitting into experimental "
                    f"& control group. Group column has name {group_col}")
        group = data[group_col].values
        return features, targets, group
    # implicit else
    return features, targets

def get_sites(filename, site_col, **kwargs):
    group_col = kwargs.get('group_col', None)
    group_value = kwargs.get('group_value', None)

    data = pd.read_csv(filename, delimiter = ';')
    data.loc[:, site_col] = data[site_col].str.rsplit('_', n=1).str[0]
    if group_col and group_value:
        data = data[data[group_col] == group_value]
    sites = data[site_col].to_list()
    return sites

def p_value_emp(arr1, arr2, permutations=100000):
    """
    Calculate the empirical p-value for the difference in means
    between two groups by performing permutation test.

    :param array-like arr1: Data for the first group.
    :param array-like arr2: Data for the second group.
    :param int permutations: Number of permutations to perform
                                 for the permutation test. Default is 100,000.

    :return: Empirically calculated p-value for the observed difference in means.
    :rtype: float
    """

    # Observed test statistic (difference in means)
    observed_stat = np.mean(arr2) - np.mean(arr1)

    # Initialize array to store permuted test statistics
    permuted_stat = np.zeros(permutations)

    # Perform permutations and calculate permuted test statistics
    for i in range(permutations):
        # Concatenate and randomly permute the data
        combined_data = np.concatenate((arr1, arr2))
        np.random.shuffle(combined_data)

        # Split permuted data into two groups
        permuted_arr1 = combined_data[:len(arr1)]
        permuted_arr2 = combined_data[len(arr1):]

        # Calculate test statistic for permuted data
        permuted_statistic = np.mean(permuted_arr2) - np.mean(permuted_arr1)

        # Store permuted statistic
        permuted_stat[i] = permuted_statistic

    # Calculate p-value
    p_value = np.sum(np.abs(permuted_stat) >= np.abs(observed_stat)) / permutations

    print("Empirical p-value:", p_value)

    return p_value


def new_prediction(features, targets, model):
    _, axa = plt.subplots(figsize=(10, 8))

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    y_pred = model.predict(features)
    mae = mean_absolute_error(targets, y_pred)
    r2 = r2_score(targets, y_pred)
    print("Mean Absolute Error on exp:", mae)
    print("R-squared on exp:", r2)

    pad_new = y_pred.ravel()-targets
    
    axa.scatter(targets, y_pred, color = 'k', alpha =0.5,
                label =f'MAE : {mae:.2} y\n$R^2$ : {r2:.2}')

    target_range = [targets.min(), targets.max()]
    # Plot the ideal line (y=x)
    axa.plot(target_range, target_range, 'k--', lw=2)

    # Set plot labels and title
    axa.set_xlabel('Actual age [y]', fontsize = 20)
    axa.set_ylabel('Predicted age [y]', fontsize = 20)
    axa.set_title('Actual vs. predicted age - ASD', fontsize = 24)

    # Add legend and grid to the plot
    axa.legend(fontsize = 16)
    axa.grid(False)
    # plt.savefig('linear_reg_exp.png', transparent = True)
    
    return pad_new, mae, r2

def sites_barplot(numbers, sites):
    numbers = np.abs(numbers)
    df = pd.DataFrame({
    'Numbers': numbers,
    'Sites': sites
    })
    grouped_df = df.groupby('Sites')['Numbers'].mean().reset_index()

    # Plot the DataFrame
    ax = grouped_df.plot.barh(legend=False, color='RoyalBlue', width=0.7, figsize=(10, 6))

    # Set the labels and title
    ax.set_xlabel('MAE values')
    ax.set_title('Bar plot of MAE across sites')
    plt.xticks(rotation=0)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.tick_params(top = "off", right = "off")

    plt.tight_layout()
    plt.savefig("../plots/bar_plot.png", transparent = True)

def mean_values_on_site(filename, targets, group, ex_cols, model):
        '''Given a neural network model and a dataset, this function calculates
        the estimators (pad, mae, r2) for each site of acquisition. The function always prints
        back some values. If extremal is True it prints the maximum and the minimum of the
        mean of the estimators across all sites, if it is False prints the mean of the
        estimators across all sites for each site. 
        First return of the function is an array containing all the pad values across every site.
        The other three return are dictionaries containing the meaned estimators as values and
        the sites of acquisition as keys  
        
        :param filename: path to the CSV file containing the dataset 
        :type filename: str
        :param targets: name of the column holding target values
        :type targets: str
        :param group: name of the column indicating the group (experimental vs control)
        :type group: str
        :param ex_cols: number of columns excluded from dataset
        :type ex_cols: int
        :param model: neural network model
        :type model: sequential
        :param extremal: printing modalities, False for detailed printings
        :type extremal: Bool
        :return: XXXXXXXX 
        
        '''
        #Reading csv file given full path
        df = pd.read_csv(filename, delimiter=';')

        #Selecting only one group (experimental)
        df = df[df[group] != -1]

        #Taking only site information in a new column
        df['SITE'] = df['FILE_ID'].str.split('_').str[0]
        
        #Vectors, dictionaries and figure initialization
        pad_ads = np.array([])
        appended_pad = np.array([])
        mean_pad = {}
        mean_mae = {}
        mean_r2 = {}

        _, ax = plt.subplots(figsize=(10, 8))

        #Ranges for the plot:
        target_range = [df[targets].values.min(), df[targets].values.max()]
        #Plot the ideal line (y=x)
        ax.plot(target_range, target_range, 'k--', lw=2)
        #Setting plot configurations
        ax.set_xlabel('Actual age [y]', fontsize = 20)
        ax.set_ylabel('Predicted age [y]', fontsize = 20)
        ax.set_title('Actual vs. predicted age - ASD', fontsize = 24)
        colormap = cmaps.get_cmap('tab20')
        colors = [colormap(i) for i,_ in enumerate(df['SITE'].values)]

        #Starting cycle over each site
        for i, (site, grouped_data) in enumerate(df.groupby('SITE'), 1):

            #Features taken form ex_cols to end-1 in order to exlude new SITE column
            features_exp = grouped_data.iloc[:, ex_cols:-1].values
            #Target extraction:
            targets_exp = grouped_data[targets].values
            #extracting values for each group:
            pad_ads, mae, r2 = new_prediction_step(features_exp, targets_exp, model, ax, color = colors[i])
            #Storing values:
            appended_pad = np.append(appended_pad, pad_ads)
            mean_pad[site] = pad_ads.mean()
            mean_mae[site] = mae.mean()
            mean_r2[site] = r2.mean()
                    
        else:
            print(mean_pad)
            print(mean_mae)
            print(mean_r2)

        return appended_pad, mean_pad, mean_mae, mean_r2

if __name__ == "__main__":

    # Uncomment for a rapid test
    
    # check_for_spurious(df, show = True)
    # check_site_correlation(df).to_csv("../data/site_correlation.csv")
    # df = handle_spurious(df, "FIQ",
    #                      "Left-vessel_Volume_mm3",
    #                      "Right-vessel_Volume_mm3",
    #                      "5th-Ventricle_Volume_mm3")
    # df.to_csv('../data/abide_clean.csv')
    # check_site_correlation(df).to_csv("site_correlation_no_spurious.csv")
    df = pd.read_csv("../data/abide.csv", delimiter = ';')
    # # handle_spurious(df)
    get_correlation(df)

