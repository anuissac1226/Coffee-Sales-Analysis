import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Review the Dataset Structure
coffee_id = pd.read_csv('data/coffee_id.csv')
coffee_clean = pd.read_csv('data/coffee_clean.csv')
coffee = pd.read_csv('data/coffee.csv')

# Display first few rows
print(coffee_id.head())
print(coffee_clean.head())
print(coffee.head())

print('\n***** Dataset Overview: coffee_id *****\n')
print(coffee_id.info())
print('\n***** Dataset Overview: coffee_clean *****\n')
print(coffee_clean.info())
print('\n***** Dataset Overview: coffee *****\n')
print(coffee.info())

# create a new column 'slug_cleaned' by removing '/review/' from each sulg entry
coffee['slug_cleaned'] = coffee['slug'].str.replace('/review/', '', regex=False)
# Rename the original 'slug' column to 'slug_old'
coffee.rename(columns={'slug': 'slug_old'}, inplace=True)
coffee.rename(columns={'slug_cleaned': 'slug'}, inplace=True)

# Check the new column with modified slug and old slug
coffee[['slug_old', 'slug']].head()

# Check for unique identifiers
print("Unique slugs in coffee_id:", coffee_id['slug'].nunique())
print("Unique slugs in coffee_clean:", coffee_clean['slug'].nunique())
print("Unique slugs in coffee:", coffee['slug'].nunique())

print("Unique name in coffee_id:", coffee_id['name'].nunique())
print("Unique name in coffee:", coffee['name'].nunique())

print("Unique review_date in coffee_id:", coffee_id['review_date'].nunique())
print("Unique review_date in coffee:", coffee['review_date'].nunique())

#Identify Data Issues:
print(coffee_id.isnull().sum())
print(coffee_clean.isnull().sum())
print(coffee.isnull().sum())

#Check duplicates
print("Duplicated slugs in coffee_id:", coffee_id[coffee_id.duplicated(subset='slug')].shape[0])
print("Duplicated slugs in coffee_clean:", coffee_clean[coffee_clean.duplicated(subset='slug')].shape[0])
print("Duplicated slugs in coffee:", coffee[coffee.duplicated(subset='slug')].shape[0])

# Check for duplicates in name
print("Duplicated names in coffee_id:", coffee_id[coffee_id.duplicated(subset='name')].shape[0])
print("Duplicated names in coffee:", coffee[coffee.duplicated(subset='name')].shape[0])

# Check for duplicates in review_date
print("Duplicated review_dates in coffee_id:", coffee_id[coffee_id.duplicated(subset='review_date')].shape[0])
print("Duplicated review_dates in coffee:", coffee[coffee.duplicated(subset='review_date')].shape[0])

print(coffee_id.duplicated().sum())
print(coffee_clean.duplicated().sum())
print(coffee.duplicated().sum())

#Inconsistencies in categorical values
print("Unique roast values (coffee):", coffee['roast'].unique())
print("Unique regions (coffee_clean):", coffee_clean[['region_africa_arabia', 'region_caribbean',
                                                     'region_central_america', 'region_hawaii',
                                                     'region_asia_pacific', 'region_south_america']].sum())
#Remove duplicate colum from coffe_cleaned
coffee_clean = coffee_clean.drop(columns=['type_with_milk.1'])
#Remove unwanted colum from coffe_cleaned
coffee_clean = coffee_clean.drop(columns=['clean_text'])
coffee_clean.info()

#Merge coffee_id with coffee_clean
merged_coffee = coffee_id.merge(
    coffee_clean,
    on='slug',
    how='left'
)
merged_coffee.info()
print(merged_coffee.head())

#Find slugs that are in coffee but not in merged_coffee
extra_in_coffee = coffee[~coffee['slug'].isin(merged_coffee['slug'])]

#Check the extra entries in merged_coffee
print("Extra entries in coffee (not in merged_coffee):")
print(extra_in_coffee[['slug']])

#Remove extra entries in coffee (those not in merged_coffee)
coffee_filtered = coffee[coffee['slug'].isin(merged_coffee['slug'])]

# Merge coffee_clean with merged_coffee while removing duplicate columns
merged_df = merged_coffee.merge(
    coffee_filtered.drop(columns=[
        'slug_old', 'name', 'rating', 'roaster', 'review_date',
        'aroma', 'body', 'flavor',
        'region_caribbean', 'region_central_america', 'region_hawaii',
        'region_asia_pacific', 'region_south_america', 'region_africa_arabia',
        'type_espresso', 'type_organic', 'type_fair_trade', 'type_decaffeinated',
        'type_pod_capsule', 'type_blend', 'type_estate','desc_1','desc_2','desc_3','desc_4','all_text'
    ]),
    on='slug',
    how='left'
)
merged_df.head()

print(merged_df.columns)

print(merged_df['rating'].unique())
#Impute missing ratings using averages by roaster
merged_df['rating'] = merged_df['rating'].replace('NR', np.nan).astype(float)
merged_df['rating'] = merged_df.groupby('roaster')['rating'].transform(lambda x: x.fillna(x.mean()))
print(merged_df['rating'].head())
merged_df['rating'].isna().sum()

# Drop rows with 'roast' column has missing values
merged_df = merged_df.dropna(subset=['roast'])
merged_df = merged_df.dropna(subset=['origin'])
#Impute with Mean
merged_df['aftertaste'] = merged_df['aftertaste'].fillna(merged_df['aftertaste'].mean())

print(merged_df.isna().sum())

#Normalize Rating(1-10)
new_min = 1
new_max = 10
# Find min and max values of the rating column
rating_min = merged_df['rating'].min()
rating_max = merged_df['rating'].max()
#Normalization to scale between 1 and 10
merged_df['normalized_rating'] = ((merged_df['rating'] - rating_min) / (rating_max - rating_min)) * (new_max - new_min) + new_min
print(merged_df[['rating', 'normalized_rating']].head())

#Extract country from origin
merged_df['origin_derived'] = merged_df['origin'].str.rsplit(',', n=1).str[-1].str.strip()
merged_df[['origin', 'origin_derived']].head()

#Add review month and year form review date
merged_df['review_date'] = pd.to_datetime(merged_df['review_date'])
merged_df['review_year'] = merged_df['review_date'].dt.year
merged_df['review_month'] = merged_df['review_date'].dt.strftime('%B')
merged_df[['review_date', 'review_year', 'review_month']].head()

def popularity(rating):
    if rating >= 8.0:
        return 'Highly Popular'
    elif rating >= 5.0:
        return 'Moderately Popular'
    else:
        return 'Less Popular'
print('Final dataset overview:')
print(merged_df.info())
print(merged_df.isna().sum())
# Add new column 'popularity_tier'
merged_df['popularity_tier'] = merged_df['normalized_rating'].apply(popularity)

#Save new dataset
merged_df.to_csv('data/merged_df.csv', index=False)




#Calculate average rating per region
region_col = {
    'region_africa_arabia','region_caribbean','region_central_america',
    'region_hawaii','region_asia_pacific','region_south_america'
}
avg_rating_by_region = {}
for i in region_col:
    avg_rating = merged_df.loc[merged_df[i] == 1, 'normalized_rating'].mean()
    avg_rating_by_region[i] = avg_rating
for reg, avg in avg_rating_by_region.items():
    print(f'avg rating of {reg} is {avg}')

#Plot average rating per region
region_df = pd.DataFrame({'Region' : list(avg_rating_by_region.keys()),'Avg_rating_region' : list(avg_rating_by_region.values())})
coffee_palette = ['saddlebrown', 'sienna', 'chocolate', 'peru', 'tan', 'wheat']
plt.figure(figsize=(8, 6))
sns.barplot(data=region_df, x='Region', y='Avg_rating_region', palette=coffee_palette)
plt.xlabel('Region')
plt.ylabel('Average Rating')
plt.title('Average Rating by Region')
plt.grid(axis='y', linestyle='--')
plt.xticks(rotation=45)
plt.show()

#Calculate average rating per roast type
roast_type_col = {
    'roast_dark','roast_light','roast_medium','roast_medium_dark','roast_medium_light','roast_very_dark'
}
avg_rating_by_roast_type = {}
for i in roast_type_col:
    avg_rating = merged_df.loc[merged_df[i] == 1, 'rating'].mean()
    avg_rating_by_roast_type[i] = avg_rating
for roast_type, avg in avg_rating_by_roast_type.items():
    print(f'avg rating of {roast_type} is {avg}')

#Plot average rating per roast type
roast_type_df = pd.DataFrame({'Roast_type' : list(avg_rating_by_roast_type.keys()),'Avg_rating_roast_type' : list(avg_rating_by_roast_type.values())})

plt.figure(figsize=(8, 6))
sns.barplot(data=roast_type_df, x='Roast_type', y='Avg_rating_roast_type',palette=coffee_palette)
plt.xlabel('Roast Type')
plt.ylabel('Average Rating')
plt.title('Average Rating by Roast Type')
plt.grid(axis='y', linestyle='--')
plt.xticks(rotation=45)
plt.show()

#top-rated products
top_rated_products = (
    merged_df[['slug', 'name', 'roaster', 'normalized_rating']]
    .sort_values(by='normalized_rating', ascending=False)
)
top_rated_10_products = top_rated_products[['name','normalized_rating']].head(10)
print("Top-Rated Products:")
print(top_rated_10_products)
#Barplot for top-rated products
plt.figure(figsize=(8, 6))
sns.barplot(x='normalized_rating', y='name', data=top_rated_10_products, palette=coffee_palette)
plt.title("Top Rated Products")
plt.xlabel("Rating")
plt.ylabel("Product Name")
plt.show()

#Most-Reviewed Roasters
most_reviewed_roasters = (merged_df.groupby('roaster').size().reset_index(name='review_count').sort_values(by='review_count', ascending=False).head(10))
print("Most-Reviewed Roasters:")
print(most_reviewed_roasters)

#Barplot for Most-Reviewed Roasters
plt.figure(figsize=(8, 6))
sns.barplot(y='roaster', x='review_count', data=most_reviewed_roasters, palette=coffee_palette)
plt.title("Most-Reviewed(Top 10) Roasters")
plt.xlabel("Count")
plt.ylabel("Roaster")
plt.show()

#Top-Rated Roasters
top_rated_roasters = (merged_df.groupby('roaster')['normalized_rating'].mean().reset_index().sort_values(by='normalized_rating', ascending=False).head(10))
print("Top-Rated Roasters:")
print(top_rated_roasters)

#Barplot for top rated Roasters
plt.figure(figsize=(8, 6))
sns.barplot(y='roaster', x='normalized_rating', data=top_rated_roasters, palette=coffee_palette)
plt.title("Top Rated Roasters")
plt.xlabel("Rating")
plt.ylabel("Roaster")
plt.show()


#top-rated product for each year
top_rated_per_year = merged_df.loc[merged_df.groupby('review_year')['normalized_rating'].idxmax()][['review_year', 'name', 'roaster', 'normalized_rating']]
print(top_rated_per_year)

#top-rated roster per year
top_roaster_per_year = merged_df.loc[merged_df.groupby('review_year')['normalized_rating'].idxmax()][['review_year', 'roaster', 'normalized_rating']]
print(top_roaster_per_year)

#top-rated roast_type per year
#Melt roast_type columns into a single column
roast_columns = ['roast_dark','roast_light','roast_medium','roast_medium_dark','roast_medium_light','roast_very_dark']
melted_df = merged_df.melt(
    id_vars=['review_year', 'normalized_rating'],
    value_vars=roast_columns,
    var_name='roast',
    value_name='is_present'
)
#Filter the rows where roast_type is present
filtered_df = melted_df[melted_df['is_present'] == 1]
avg_rating_per_roast = filtered_df.groupby(['review_year', 'roast'])['normalized_rating'].mean().reset_index()
top_rated_roast_per_year = avg_rating_per_roast.loc[avg_rating_per_roast.groupby('review_year')['normalized_rating'].idxmax()]
print('Top rated roast type per year:\n',top_rated_roast_per_year[['review_year', 'roast', 'normalized_rating']])

#Bar plot for top-rated roast_type per year
roast_colors = {
    'roast_dark': 'saddlebrown',
    'roast_light': 'sienna',
    'roast_medium': 'chocolate',
    'roast_medium_dark': 'peru',
    'roast_medium_light': 'tan',
    'roast_very_dark': 'wheat'
}
top_rated_roast_per_year['color'] = top_rated_roast_per_year['roast'].map(roast_colors)
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=top_rated_roast_per_year,
    x='review_year',
    y='normalized_rating',
    palette=top_rated_roast_per_year['color'].tolist()
)
custom_legand = [plt.Rectangle((0,0),1,1, color=color) for color in roast_colors.values()]
plt.legend(custom_legand, roast_colors.keys(), title="Roast Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Top-Rated Roast Type Per Year')
plt.xlabel('Review Year')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--')
plt.xticks(rotation=90)
plt.show()

#top-rated region per year
#Melt roast_type columns into a single column
region_col = ['region_africa_arabia', 'region_caribbean', 'region_central_america',
               'region_hawaii', 'region_asia_pacific', 'region_south_america']
melted_df = merged_df.melt(
    id_vars=['review_year', 'normalized_rating'],
    value_vars=region_col,
    var_name='region',
    value_name='is_present'
)
filtered_df = melted_df[melted_df['is_present'] == 1]
avg_rating_per_region = filtered_df.groupby(['review_year', 'region'])['normalized_rating'].mean().reset_index()
top_rated_region_per_year = avg_rating_per_region.loc[avg_rating_per_region.groupby('review_year')['normalized_rating'].idxmax()]
print('Top rated region type per year:\n',top_rated_region_per_year[['review_year', 'region', 'normalized_rating']])

#Bar plot for top-rated region per year
region_colors = {
    'region_africa_arabia': 'burlywood',
    'region_caribbean': 'rosybrown',
    'region_central_america': 'peru',
    'region_hawaii': 'saddlebrown',
    'region_asia_pacific': 'chocolate',
    'region_south_america': 'maroon'
}
top_rated_region_per_year['color'] = top_rated_region_per_year['region'].map(region_colors)
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=top_rated_region_per_year,
    x='review_year',
    y='normalized_rating',
    palette=top_rated_region_per_year['color'].tolist()
)
custom_legand = [plt.Rectangle((0,0),1,1, color=color) for color in region_colors.values()]
plt.legend(custom_legand, region_colors.keys(), title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Top-Rated Region Per Year')
plt.xlabel('Review Year')
plt.ylabel('Average Rating')
plt.grid(axis='y', linestyle='--')
plt.xticks(rotation=90)
plt.show()


#Regional Performance
#Average rating for each roast_type across each region
region_columns = ['region_africa_arabia', 'region_caribbean', 'region_central_america',
                  'region_hawaii', 'region_asia_pacific', 'region_south_america']
roast_columns = ['roast_dark', 'roast_light', 'roast_medium', 'roast_medium_dark', 'roast_medium_light']

#Melt the region columns into a single column
melted_regions = merged_df.melt(
    id_vars=roast_columns + ['normalized_rating'],
    value_vars=region_columns,
    var_name='region',
    value_name='region_is_present'
)
# Melt roast_tpye columns
melted_roasts = melted_regions.melt(
    id_vars=['region', 'region_is_present', 'normalized_rating'],
    value_vars=roast_columns,
    var_name='roast',
    value_name='roast_is_present'
)
#Filter rows where both region and roast are present
filtered_data = melted_roasts[(melted_roasts['region_is_present'] == 1) & (melted_roasts['roast_is_present'] == 1)]
avg_rating_per_roast_region = filtered_data.groupby(['region', 'roast'])['normalized_rating'].mean().reset_index()
print('Average rating for each roast_type across each region: \n',avg_rating_per_roast_region)
#avg_rating_per_roast_region.to_csv('avg_rating_per_roast_region.csv', index=False)

#Bar plot for average rating for each roast_type across each region
region_palette = ['burlywood', 'rosybrown', 'peru', 'saddlebrown', 'chocolate', 'maroon']

plt.figure(figsize=(12, 6))
sns.barplot(data=avg_rating_per_roast_region, x='roast', y='normalized_rating', hue='region',palette=region_palette)
plt.title('Average Rating for Each Roast Type Across Each Region')
plt.xlabel('Roast Type')
plt.ylabel('Average Rating')
plt.legend(title='Region', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.show()


#Top-rated roast_type for each region
best_roast_per_region = avg_rating_per_roast_region.loc[avg_rating_per_roast_region.groupby('region')['normalized_rating'].idxmax()]
print('Top-rated roast_type for each region:\n',best_roast_per_region[['region', 'roast', 'normalized_rating']])

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=best_roast_per_region, x='region', y='normalized_rating', palette=region_palette)
bar_position = [p.get_x() + p.get_width() / 2 for p in ax.patches]
#add labels to each bar
for pos, (index, row) in zip(bar_position, best_roast_per_region.iterrows()):
    ax.text(pos, row['normalized_rating'] + 0.02, row['roast'],
            ha='center', va='bottom', fontsize=10, color='black', fontweight='light')
plt.title('Top-rated Roast Type for Each Region')
plt.xlabel('Region')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()



#Top-rated region for each roast type
#region with the highest average rating for each roast type
top_rated_region_per_roast = avg_rating_per_roast_region.loc[avg_rating_per_roast_region.groupby('roast')['normalized_rating'].idxmax()]
print('Top-rated region for each roast type:\n', top_rated_region_per_roast)

#Barplot for Top-rated region for each roast type
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=top_rated_region_per_roast, x='roast', y='normalized_rating', palette=region_palette)
bar_position = [p.get_x() + p.get_width() / 2 for p in ax.patches]
#add labels to each bar
for pos, (index, row) in zip(bar_position, top_rated_region_per_roast.iterrows()):
    ax.text(pos, row['normalized_rating'] + 0.02, row['region'],
            ha='center', va='bottom', fontsize=10, color='black', fontweight='light')
plt.title('Top-rated region for each roast type')
plt.xlabel('Roast Type')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.show()


#Most popular roast_types by review count
roast_type_col = ['roast_dark', 'roast_light', 'roast_medium', 'roast_medium_dark', 'roast_medium_light', 'roast_very_dark']
#count occurrences of each roast_type
roast_type_popularity = {}
for i in roast_type_col:
    roast_type_popularity[i] = merged_df[i].sum()
#convert into a dataframe
roast_df = pd.DataFrame({
    'Roast Type': list(roast_type_popularity.keys()),
    'Count': list(roast_type_popularity.values())
})
most_popular_roast = roast_df.loc[roast_df['Count'].idxmax(), 'Roast Type']
print(f"Most popular roast type is: {most_popular_roast}")
#Distribustion of most popular roast type accross region
region_col = ['region_africa_arabia', 'region_caribbean', 'region_central_america',
               'region_hawaii', 'region_asia_pacific', 'region_south_america']
popular_roast_region_distribution = []
for j in region_col:
    count = merged_df.loc[(merged_df[j] == 1) & (merged_df[most_popular_roast] == 1)].shape[0]
    avg_rating = merged_df.loc[(merged_df[j] == 1) & (merged_df[most_popular_roast] == 1), 'normalized_rating'].mean()
    popular_roast_region_distribution.append({'Roast_Type': most_popular_roast, 'Region': j, 'Review Count': count, 'Avg_Rating': avg_rating})
popular_roast_type_by_region = pd.DataFrame(popular_roast_region_distribution)
print('Distribution of most popular roast type accross region: \n',popular_roast_type_by_region)

#Barplot for popularity of roast_type (based od on review count)
plt.figure(figsize=(8, 6))
sns.barplot(data=roast_df, x='Roast Type', y='Count', palette=coffee_palette)
plt.title('Roast Type Preference based on Review Count')
plt.xlabel('Roast Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

#Barplot for distribution of most popular roast type(based on review count) across regions
plt.figure(figsize=(8, 6))
sns.barplot(data=popular_roast_type_by_region, x='Region', y='Avg_Rating', palette=region_palette)
plt.title(f'Distribution of {most_popular_roast} Across Regions')
plt.xlabel('Region')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

#Most popular roast_types by rating
roast_type_col = ['roast_dark', 'roast_light', 'roast_medium', 'roast_medium_dark', 'roast_medium_light', 'roast_very_dark']
#average rating for each roast type
roast_type_rating = {}
for i in roast_type_col:
    avg_rating = merged_df.loc[merged_df[i] == 1, 'normalized_rating'].mean()
    roast_type_rating[i] = avg_rating
#Convert into dataframe
roast_by_rating_df = pd.DataFrame({
    'Roast Type': list(roast_type_rating.keys()),
    'Average Rating': list(roast_type_rating.values())
})
most_popular_roast_by_rating = roast_by_rating_df.loc[roast_by_rating_df['Average Rating'].idxmax(), 'Roast Type']
print(f"Most popular roast type by rating is: {most_popular_roast_by_rating}")
#Distribustion of most popular roast type accross region
region_col = ['region_africa_arabia', 'region_caribbean', 'region_central_america',
               'region_hawaii', 'region_asia_pacific', 'region_south_america']
popular_roast_by_region_distribution = []
for j in region_col:
    count = merged_df.loc[(merged_df[j] == 1) & (merged_df[most_popular_roast_by_rating] == 1)].shape[0]
    avg_rating = merged_df.loc[(merged_df[j] == 1) & (merged_df[most_popular_roast_by_rating] == 1), 'normalized_rating'].mean()
    popular_roast_by_region_distribution.append({'Roast_Type': most_popular_roast_by_rating, 'Region': j, 'Review Count': count, 'Avg_Rating': avg_rating})
popular_roast_by_region = pd.DataFrame(popular_roast_by_region_distribution)
print('Distribution of most popular roast type accross region: \n',popular_roast_by_region)

#Barplot for popularity of roast_type (based od on rating)
plt.figure(figsize=(8, 6))
sns.barplot(data=roast_by_rating_df, x='Roast Type', y='Average Rating', palette=coffee_palette)
plt.title('Roast Type Preference Based on Rating')
plt.xlabel('Roast Type')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

#Barplot for Most Popular Roast Type(based on rating) Across Regions
plt.figure(figsize=(8, 6))
sns.barplot(data=popular_roast_by_region, x='Region', y='Avg_Rating', palette=region_palette)
plt.title(f'Distribution of {most_popular_roast_by_rating} Across Regions')
plt.xlabel('Region')
plt.ylabel('AvRaerage ting')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()


#Distribution(review count) of All Roast Types across each Region
region_col = ['region_africa_arabia', 'region_caribbean', 'region_central_america',
               'region_hawaii', 'region_asia_pacific', 'region_south_america']
roast_type_region_distribution = []
for i in roast_type_col:
    for j in region_col:
        count = merged_df.loc[(merged_df[i] == 1) & (merged_df[j] == 1)].shape[0]
        roast_type_region_distribution.append({'Roast_Type': i, 'Region': j, 'Count': count})
roast_type_by_region = pd.DataFrame(roast_type_region_distribution)
print('Distribution(review count) of All Roast Types across each Region:\n',roast_type_by_region)

#Heatmap for roast types by region based on review count
pivot_distribution = roast_type_by_region.pivot(index='Roast_Type', columns='Region', values='Count')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_distribution, annot=True, fmt='.0f', cmap='YlOrBr', cbar_kws={'label':'count'},linewidths=0.5)
plt.title('Roast Types by Region based on Review Count')
plt.xlabel('Region')
plt.ylabel('Roast Type')
plt.show()

#Relationship between sensory attributes and rating
sensory_attributes = ['aroma', 'flavor', 'acid_or_milk', 'body', 'aftertaste','normalized_rating']
#correlation matrix
correlation_matrix = merged_df[sensory_attributes].corr()
#heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrBr', fmt='.2f',  cbar_kws={'label':'Corelation Coeficient'},linewidths=0.5)
plt.title('Correlation Analysis Between Sensory Attributes and rating')
plt.xlabel('Sensory Attributes')
plt.ylabel('Sensory Attributes')
plt.tight_layout()
plt.show()