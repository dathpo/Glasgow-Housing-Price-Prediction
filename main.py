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

data_folder = Path("data/")
cols_2excl = ['CLASS', 'STNO', 'STnu', 'FLATPOSN', 'YEAR OF SALE (BUSINESS)', 'MONTH AND YEAR',
              'QUARTER_(CALENDAR)', 'OMIT OR USE']


def concatenate(folder):
    file_list = folder.glob("*.csv")
    df_list = []
    for file_name in file_list:
        df = pd.read_csv(file_name)
        clean_df = df[:-1]
        df_list.append(clean_df)
    cc_df = pd.concat(df_list, axis=0)
    cc_df.drop(columns=cols_2excl, inplace=True)
    return cc_df


dataframe = concatenate(data_folder)
dataframe.to_csv(path_or_buf='data/dataframe.csv', index=False)