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


def merge_df(simd_df, left_df, column, year, first):
    new_col = ''.join(column + '_' + str(year))
    right_df = simd_df[['Datazone', new_col]]
    right_df.columns = ['datazone', column]
    right_df['sale_year'] = year
    merged_df = pd.merge(left_df, right_df, how='left', on=['datazone', 'sale_year'])
    dupl_col_x = ''.join(column + '_x')
    dupl_col_y = ''.join(column + '_y')
    if not first:
        merged_df[column] = merged_df[dupl_col_x].fillna(merged_df[dupl_col_y])
        merged_df.drop(columns=[dupl_col_x, dupl_col_y], inplace=True)
    return merged_df

dataframe = concatenate(prices_folder)
datazone_df = from_xls()

merged = pd.merge(dataframe, datazone_df, on='postcode')

overall_simd_df = pd.read_csv('data/simd/simd-overall-2004-2012-glasgow-v2.csv')
employ_simd_df = pd.read_csv('data/simd/simd-employment-2004-2012-glasgow-v2.csv')
income_simd_df = pd.read_csv('data/simd/simd-income-2004-2012-glasgow-v2.csv')
health_simd_df = pd.read_csv('data/simd/simd-health-2004-2012-glasgow-v2.csv')
edu_simd_df = pd.read_csv('data/simd/simd-education-2004-2012-glasgow-v2.csv')
geo_simd_df = pd.read_csv('data/simd/simd-geographic-access-2004-2012-glasgow-v2.csv')
housing_simd_df = pd.read_csv('data/simd/simd-housing-2004-2012-glasgow-v2.csv')
crime_simd_df = pd.read_csv('data/simd/simd-crime-2006-2012-glasgow-v2.csv')

overall_col = 'overall_deprivation_rank'
overall_04 = merge_df(overall_simd_df, merged, overall_col, 2004, True)
overall_06 = merge_df(overall_simd_df, overall_04, overall_col, 2006, False)
overall_09 = merge_df(overall_simd_df, overall_06, overall_col, 2009, False)
overall_12 = merge_df(overall_simd_df, overall_09, overall_col, 2012, False)

empl_col = 'employment_deprivation_rank'
empl_04 = merge_df(employ_simd_df, overall_12, empl_col, 2004, True)
empl_06 = merge_df(employ_simd_df, empl_04, empl_col, 2006, False)
empl_09 = merge_df(employ_simd_df, empl_06, empl_col, 2009, False)
empl_12 = merge_df(employ_simd_df, empl_09, empl_col, 2012, False)

income_col = 'income_deprivation_rank'
income_04 = merge_df(income_simd_df, empl_12, income_col, 2004, True)
income_06 = merge_df(income_simd_df, income_04, income_col, 2006, False)
income_09 = merge_df(income_simd_df, income_06, income_col, 2009, False)
income_12 = merge_df(income_simd_df, income_09, income_col, 2012, False)

health_col = 'health_deprivation_rank'
health_04 = merge_df(health_simd_df, income_12, health_col, 2004, True)
health_06 = merge_df(health_simd_df, health_04, health_col, 2006, False)
health_09 = merge_df(health_simd_df, health_06, health_col, 2009, False)
health_12 = merge_df(health_simd_df, health_09, health_col, 2012, False)

edu_col = 'education_deprivation_rank'
edu_04 = merge_df(edu_simd_df, health_12, edu_col, 2004, True)
edu_06 = merge_df(edu_simd_df, edu_04, edu_col, 2006, False)
edu_09 = merge_df(edu_simd_df, edu_06, edu_col, 2009, False)
edu_12 = merge_df(edu_simd_df, edu_09, edu_col, 2012, False)

geo_col = 'geographic_access_deprivation_rank'
geo_04 = merge_df(geo_simd_df, edu_12, geo_col, 2004, True)
geo_06 = merge_df(geo_simd_df, geo_04, geo_col, 2006, False)
geo_09 = merge_df(geo_simd_df, geo_06, geo_col, 2009, False)
geo_12 = merge_df(geo_simd_df, geo_09, geo_col, 2012, False)

housing_col = 'housing_deprivation_rank'
housing_04 = merge_df(housing_simd_df, geo_12, housing_col, 2004, True)
housing_06 = merge_df(housing_simd_df, housing_04, housing_col, 2006, False)
housing_09 = merge_df(housing_simd_df, housing_06, housing_col, 2009, False)
housing_12 = merge_df(housing_simd_df, housing_09, housing_col, 2012, False)

crime_col = 'crime_deprivation_rank'
crime_06 = merge_df(crime_simd_df, housing_12, crime_col, 2006, True)
crime_09 = merge_df(crime_simd_df, crime_06, crime_col, 2009, False)
crime_12 = merge_df(crime_simd_df, crime_09, crime_col, 2012, False)

df = crime_12
df = df.sort_values(by=['sale_date', 'july_2013_price'], ascending=False)
df = df[['street', 'postcode', 'datazone', 'local_housing_forum', overall_col, empl_col, income_col,
                     health_col, edu_col, geo_col, housing_col, crime_col, 'sale_date', 'sale_quarter',
                     'sale_year', 'sale_month', 'nominal_price', 'july_2013_price', 'build', 'buyer_origin']]
df.to_csv(path_or_buf='dataframe.csv', index=False)

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