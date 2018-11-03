from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import sklearn
import seaborn as sns
import geopandas as gpd

from sklearn import datasets, cluster, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale, StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# COMMENT CODE FOR ASSIGNMENT!

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
prices_folder = Path("data/")
cols_2excl = ['CLASS', 'STNO', 'STnu', 'FLATPOSN', 'YEAR OF SALE (BUSINESS)', 'RPI', 'DEFLATOR', 'OMIT OR USE']
new_cols = ['street', 'postcode', 'sale_month', 'sale_year', 'sale_date', 'sale_quarter', 'nominal_price',
            'july_2013_price', 'buyer_origin', 'build', 'local_housing_forum']

fp = "/home/david/Desktop/Personal/University/CS982/Assignment/alternative_data/2011_Census_Glasgow_City_(shp)/DZ_2011_EoR_Glasgow_City.shp"
map_df = gpd.read_file(fp)

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

def create_pdf_fig(filename):
    pwd = os.path.abspath(os.path.dirname(sys.argv[0]))
    graph_path = os.path.join(pwd, '{}.pdf'.format(filename))
    pdf = PdfPages(graph_path)
    plt.savefig(pdf, format='pdf', bbox_inches='tight')
    plt.show()
    pdf.close()
    pdf = None

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

df.local_housing_forum = df.local_housing_forum.str.replace(' ', '_')
df = pd.get_dummies(df, columns=['build'], drop_first=True)
df = pd.get_dummies(df, columns=['local_housing_forum'])

df = df[pd.notnull(df['crime_deprivation_rank'])]               # Drop NaN rows, keeps 2006, 2009, 2012 only
df.drop(columns='nominal_price', inplace=True)

df = df.rename(index=str, columns={"build_RESALE": "resale",
                                      "local_housing_forum_Baillieston_Shettleston": "Baillieston_Shettleston",
                                      "local_housing_forum_Drumchapel_Anniesland_and_Garscadden_Scotstounhill": "Drumchapel_Anniesland_Garscadden_Scotstounhill",
                                      "local_housing_forum_East_Centre_and_Calton": "East_Centre_Calton",
                                      "local_housing_forum_Central_and_West": "Central_West",
                                      "local_housing_forum_Govan_and_Craigton": "Govan_Craigton",
                                      "local_housing_forum_Greater_Pollok_and_Newlands_Auldburn": "Greater_Pollok_Newlands_Auldburn",
                                      "local_housing_forum_Langside_and_Linn": "Langside_Linn",
                                      "local_housing_forum_Maryhill_Kelvin_and_Canal": "Maryhill_Kelvin_Canal",
                                      "local_housing_forum_Pollokshields_and_Southside_Central": "Pollokshields_Southside_Central",
                                      "local_housing_forum_Springburn": "Springburn",
                                      "sale_month": "month", "sale_year": "year", "sale_quarter":"quarter", "sale_date":"date",
                                      "crime_deprivation_rank":"crime", "housing_deprivation_rank":"housing",
                                      "geographic_access_deprivation_rank":"geographic_access", "education_deprivation_rank":"education",
                                      "health_deprivation_rank":"health", "income_deprivation_rank":"income",
                                      "employment_deprivation_rank":"employment", "overall_deprivation_rank":"overall",
                                      "july_2013_price":"price"})

df_for_map = df[['datazone','price','overall','employment','income','health','education','geographic_access','housing','crime','year','month','resale']]
merged_map = map_df.set_index('GSS_CODEDZ').join(df_for_map.set_index('datazone'))


variable = 'price'
vmin, vmax = 0, 1
# create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(10, 6))
merged_map.plot(column=variable, cmap='YlGn', ax=ax,
            linewidth=0.1, edgecolor='0.5', scheme='equal_interval',
            k=15)
ax.axis('off')
ax.set_title('Glasgow Overall SIMD', \
              fontdict={'fontsize': '25',
                        'fontweight' : '3'})
# create an annotation for the  data source
ax.annotate('Source: Glasgow Open Data, 2014',
           xy=(0.1, .08), xycoords='figure fraction',
           horizontalalignment='left', verticalalignment='top',
           fontsize=10, color='#555555')
# Create colorbar as a legend
sm = plt.cm.ScalarMappable(cmap='YlGn')
sm._A = []
cbar = fig.colorbar(sm)
# this will save the figure as a high-res png. you can also save as svg
# fig.savefig('testmap.png', dpi=800)


df.datazone = df.datazone.str.replace(r'S\d{5}', r'', regex=True)
df.datazone = df.datazone.apply(pd.to_numeric)

df.overall = df.overall/6505
df.employment = df.employment/6505
df.income = df.income/6505
df.health = df.health/6505
df.education = df.education/6505
df.geographic_access = df.geographic_access/6505
df.housing = df.housing/6505
df.crime = df.crime/6505

print("Find most important features relative to target")
# df = df.pivot(columns='overall', values='price')
corr = df.corr()
fig = plt.figure()
overall1 = sns.heatmap(corr)
fig.savefig('overall1.pdf', bbox_inches="tight")
corr.sort_values(["overall"], ascending = False, inplace = True)
print(corr.overall)

df = df[['price','datazone','overall','employment','income','health','education','geographic_access','housing','crime',
         'month','resale','buyer_origin']]

print()
corr = df.corr()
fig = plt.figure()
overall2 = sns.heatmap(corr)
corr.sort_values(["overall"], ascending = False, inplace = True)
fig.savefig('overall2.pdf', bbox_inches="tight")
print(corr.overall)
print()

# print(df.buyer_origin)
print(df.shape)
# df = pd.get_dummies(df, columns=['buyer_origin'])
# df = pd.get_dummies(df, columns=['datazone'])
le = LabelEncoder()
df['buyer_origin'] = le.fit_transform(df['buyer_origin'].astype(str))
print(df.buyer_origin)
print()
print(df.shape)

target = df.values[:, 0]
attributes = df.values[:, 1:]

# scaled = scale(attributes)
scaler = StandardScaler()
scaler.fit(attributes)
scaled_data = scaler.transform(attributes)

pca = PCA(n_components=12)
pca.fit(scaled_data)

var_ratio = pca.explained_variance_ratio_.cumsum()
print(var_ratio)
# create_pdf_fig('var_ratio')
plt.plot(var_ratio)
plt.savefig('sample.pdf')
print(pca.singular_values_)


n_samples, n_features = scaled_data.shape
n_digits = len(np.unique(target))
print(n_digits)
# encoded_target = LabelEncoder().fit_transform(target)
# model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage="ward", affinity="euclidean")
# model.fit(scaled_data)

# print(encoded_outcome)
print("length:", n_digits)
print("n samples:", n_samples, "n features:", n_features)
print()
# print("Model Labels:", model.labels_)
print()
# print("Silhoutte Score:", metrics.silhouette_score(scaled_data, model.labels_))
# print("Homogeneity Score:", metrics.homogeneity_score(target, model.labels_))
# print("Completeness Score:", metrics.completeness_score(target, model.labels_))

model = linkage(scaled_data, 'ward')
fig = plt.figure()
plt.figure()
plt.title('Hierarchical Clustering Dendrogram (Ward)')
plt.xlabel('Sample Index')
plt.ylabel('Distance (Euclidean)')
dendrogram(
        model,
        leaf_rotation=90.,
        leaf_font_size=10.,
    )
fig.savefig('dendro_ward_eu.pdf', bbox_inches="tight")

# df.build = df.build.str.replace('RESALE', '0')
# df.build = df.build.str.replace('NEW', '1')
# df.build = df.build.apply(pd.to_numeric)

# print("origin:", len(df.buyer_origin.unique()))
# print("streets:", len(df.street.unique()))
# print("datazones:", len(df.datazone.unique()))

# df.to_csv(path_or_buf='dataframe.csv', index=False)

# print(df.price.describe().apply(lambda x: format(x, '.2f')).to_latex())
# print()
# print(df.info())
# print()
# print(df.head())
# print()

# df.overall.plot(kind='box', subplots=True)
# create_pdf_fig('simd_box')
#
# df.price.plot(kind='box')
# create_pdf_fig('price_box')


# def hierarchical_clustering():
#     target = df["price"]
#     # print(vg_sales.as_matrix()[2068])
#     # print(vg_sales.as_matrix()[3568])
#     df.drop("Name") # Pop name as it should not be analysed
#     df.drop("NA_Sales")
#     vg_sales.pop("EU_Sales")
#     vg_sales.pop("JP_Sales")
#     vg_sales.pop("Other_Sales")
#     vg_sales.pop("Developer")
#     # vg_sales.pop("Platform")
#     vg_sales.pop("Publisher")
#     # vg_sales.pop("Rating")
#     # vg_sales.pop("Genre")
#
#     cols_to_transform = ['Platform', 'Rating', 'Genre']
#     vg_sales_dummies = pd.get_dummies(data=vg_sales, columns=cols_to_transform)
#
#     data = scale(vg_sales_dummies)
#
#     n_digits = len(np.unique(target))
#     model = cluster.AgglomerativeClustering(n_clusters=n_digits, linkage='average', affinity ='cosine')
#     model.fit(data)
#
#     model = linkage(data, 'ward')
#     plt.figure()
#     plt.title('Hierarchical Clustering Dendrogram (Ward)')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Distance')
#
#     # dendrogram(
#     #     model, truncate_mode='lastp',  # show only the last p merged clusters
#     #     p=50,  # show only the last p merged clusters
#     #     show_leaf_counts=False,  # otherwise numbers in brackets are counts
#     #     leaf_rotation=90.,
#     #     leaf_font_size=10.,
#     #     show_contracted=True,  # to get a distribution impression in truncated branches
#     # )
#
#     dendrogram(
#         model,
#         leaf_rotation=90.,
#         leaf_font_size=10.,
#     )
#
#     plt.show()
#     print("show")