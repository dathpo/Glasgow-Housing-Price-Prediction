from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
pd.set_option('display.max_columns', None)
prices_folder = Path("data/")
cols_2excl = ['CLASS', 'STNO', 'STnu', 'FLATPOSN', 'YEAR OF SALE (BUSINESS)', 'RPI', 'DEFLATOR', 'OMIT OR USE']
new_cols = ['street', 'postcode', 'sale_month', 'sale_year', 'sale_date', 'sale_quarter', 'nominal_price',
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
merged_12 = merged_12[['street', 'postcode', 'datazone', 'local_housing_forum', 'overall_deprivation_rank',
                       'sale_date', 'sale_quarter', 'sale_year', 'sale_month', 'nominal_price', 'july_2013_price', 'build', 'buyer_origin']]
# merged_12.to_csv(path_or_buf='dataframe.csv', index=False)
df = merged_12

# rs_data = np.array(rs_fitness)
# hs_internal = np.array(hc_internal_fitness)
# hs_external = np.array(hc_external_fitness)
# ga_data = np.array(ga_fitness)

# test_cases_per_test_suite = np.array([5, 10, 20, 23, 30, 50, 100])
# unique_large_apfd = np.array([0.4594736842105263, 0.6063157894736844, 0.6867105263157895, 0.6978260869565216, 0.7128947368421051, 0.7326842105263159, 0.7480263157894737])
# full_large_apfd = np.array([0.44631578947368417, 0.6023684210526316, 0.6846052631578947, 0.6958810068649884, 0.7122807017543858, 0.7320526315789474, 0.7476578947368421])

# plt.plot(test_cases_per_test_suite, unique_large_apfd, '-gD')
# plt.xlabel("Test Cases per Test Suite")
# plt.ylabel("Mean Fitness (APFD)")
# plt.xticks(np.arange(min(test_cases_per_test_suite), max(test_cases_per_test_suite) + 1, 5.0))

## combine these different collections into a list
# data_to_plot = [rs_data, hs_internal, hs_external, ga_data]

# Create a figure instance
# fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
# ax = fig.add_subplot(111)

## add patch_artist=True option to ax.boxplot()
# bp = ax.boxplot(data_to_plot, patch_artist=True)

## change outline color, fill color and linewidth of the boxes
# for box in bp['boxes']:
#     # change outline color
#     box.set(color='#7570b3', linewidth=2)
#     # change fill color
#     box.set(facecolor='#1b9e77')

## change color and linewidth of the whiskers
# for whisker in bp['whiskers']:
#     whisker.set(color='#7570b3', linewidth=2)

## change color and linewidth of the caps
# for cap in bp['caps']:
#     cap.set(color='#7570b3', linewidth=2)

## change color and linewidth of the medians
# for median in bp['medians']:
#     median.set(color='#b2df8a', linewidth=2)

## change the style of fliers and their fill
# for flier in bp['fliers']:
#     flier.set(marker='o', color='#e7298a', alpha=0.5)

## Custom x-axis labels
# ax.set_xticklabels(['Random Search', 'HC Internal Swap', 'HC External Swap', 'Genetic Algorithm'])

## Remove top axes and right axes ticks
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()

# Save the figure
def create_pdf_fig(filename):
    pwd = os.path.abspath(os.path.dirname(__file__))
    graph_path = os.path.join(pwd, '{}.pdf'.format(filename))
    pdf = PdfPages(graph_path)
    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.show()
    pdf.close()
    pdf = None

print(df.july_2013_price.describe().apply(lambda x: format(x, '.2f')).to_latex())
print()
print(df.info())
print()
print(df.head())
print()
print("Find most important features relative to target")
corr = df.corr()
corr.sort_values(["july_2013_price"], ascending = False, inplace = True)
print(corr.july_2013_price)

df.overall_deprivation_rank.plot(kind='box', subplots=True)
create_pdf_fig('simd_box')

df.july_2013_price.plot(kind='box')
create_pdf_fig('price_box')