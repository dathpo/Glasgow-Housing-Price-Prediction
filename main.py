from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn

# from sklearn import datasets, cluster, metrics
# from sklearn.metrics import r2_score
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import scale
# from sklearn.cluster import KMeans
# from scipy.cluster.hierarchy import dendrogram, linkage

# COMMENT CODE FOR ASSIGNMENT!

pd.options.mode.chained_assignment = None
prices_folder = Path("data/")
cols_2excl = ['CLASS', 'STNO', 'STnu', 'FLATPOSN', 'YEAR OF SALE (BUSINESS)',
              'MONTH OF SALE', 'QUARTER_(CALENDAR)', 'OMIT OR USE']
new_cols = ['street', 'postcode', 'sale_year', 'sale_date', 'nominal_price', 'retail_price_index', 'deflator',
            'july_2013_price', 'buyer_origin', 'build', 'local_housing_forum']


def concatenate(folder):
    file_list = folder.glob("*.csv")  # Find .csv files
    df_list = []
    for file_name in file_list:
        single_df = pd.read_csv(file_name, low_memory=False)
        clean_df = single_df[:-1]  # Remove last row of each CSV file as it's not relevant
        df_list.append(clean_df)
    df = pd.concat(df_list, axis=0, sort=False)  # Concatenate CSV files
    df.drop(columns=cols_2excl, inplace=True)  # Drop useless columns
    df.columns = new_cols  # Rename columns with code-friendly terms
    # Regular Expression to remove symbol prefix for some cells in price column
    # df.july_2013_price = df.july_2013_price.str.replace(r'[^0-9.]+(?=\d+)', r'', regex=True)
    # Regular Expression to convert date formats from YYYY-MM-DD to YYYY-MM as DD was always 01 anyway
    df.sale_date = df.sale_date.str.replace(r'-\d{2}$', r'', regex=True)
    # Regular Expression to convert date formats from DD/MM/YYYY to YYYY-MM
    df.sale_date = df.sale_date.str.replace(r'\d{2}\/(\d{2})\/(\d{4})$', r'\2-\1', regex=True)
    # print(df.iloc[96165]['july_2013_price']) # 2000
    return df


def from_xls():
    xls = pd.ExcelFile('data/datazones.xls')
    dz_df = pd.read_excel(xls, 'Sheet1')
    dz_df = dz_df[['Postcode', 'DataZone']]
    dz_df.columns = ['postcode', 'datazone']
    return dz_df


dataframe = concatenate(prices_folder)
datazone_df = from_xls()
merged = pd.merge(dataframe, datazone_df, on='postcode')

simd_df = pd.read_csv('data/simd/simd-overall-2004-2012-glasgow-v2.csv')
simd_df_04 = simd_df[['Datazone', 'overall_deprivation_rank_2004']]
simd_df_04.columns = ['datazone', 'overall_deprivation_rank']
simd_df_04['sale_year'] = 2004
merged_04 = pd.merge(merged, simd_df_04, how='left', on=['datazone', 'sale_year'])

simd_df_06 = simd_df[['Datazone', 'overall_deprivation_rank_2006']]
simd_df_06.columns = ['datazone', 'overall_deprivation_rank']
simd_df_06['sale_year'] = 2006
merged_06 = pd.merge(merged_04, simd_df_06, how='left', on=['datazone', 'sale_year'])
merged_06['overall_deprivation_rank'] = merged_06['overall_deprivation_rank_x'].fillna(merged_06['overall_deprivation_rank_y'])
merged_06.drop(columns=['overall_deprivation_rank_x', 'overall_deprivation_rank_y'], inplace=True)

simd_df_09 = simd_df[['Datazone', 'overall_deprivation_rank_2009']]
simd_df_09.columns = ['datazone', 'overall_deprivation_rank']
simd_df_09['sale_year'] = 2009
merged_09 = pd.merge(merged_06, simd_df_09, how='left', on=['datazone', 'sale_year'])
merged_09['overall_deprivation_rank'] = merged_09['overall_deprivation_rank_x'].fillna(merged_09['overall_deprivation_rank_y'])
merged_09.drop(columns=['overall_deprivation_rank_x', 'overall_deprivation_rank_y'], inplace=True)

simd_df_12 = simd_df[['Datazone', 'overall_deprivation_rank_2012']]
simd_df_12.columns = ['datazone', 'overall_deprivation_rank']
simd_df_12['sale_year'] = 2012
merged_12 = pd.merge(merged_09, simd_df_12, how='left', on=['datazone', 'sale_year'])
merged_12['overall_deprivation_rank'] = merged_12['overall_deprivation_rank_x'].fillna(merged_12['overall_deprivation_rank_y'])
merged_12.drop(columns=['overall_deprivation_rank_x', 'overall_deprivation_rank_y'], inplace=True)
merged_12 = merged_12.sort_values(by=['sale_date', 'july_2013_price'], ascending=False)
merged_12.to_csv(path_or_buf='dataframe.csv', index=False)