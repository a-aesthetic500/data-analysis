import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv("movies.csv")
print(df.info())

# PART 1: DATA EXPLORATION & CLEANING

# 1.1 Initial exploration
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Data types:\n", df.dtypes)

# 1.2 Handle missing metascore (fill median by genre)
df['metascore'] = df.groupby('genre')['metascore'].transform(lambda x: x.fillna(x.median()))

# 1.3
df = df[df['budget'] >= 0]      
df = df[df['revenue'] >= 0]      
df = df[df['runtime'] >= 10]    

def find_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    return outliers

outliers_budget = find_outliers('budget')
outliers_revenue = find_outliers('revenue')
outliers_runtime = find_outliers('runtime')
outliers_votes = find_outliers('votes')

print("Number of budget outliers:", len(outliers_budget))
print("Number of revenue outliers:", len(outliers_revenue))
print("Number of runtime outliers:", len(outliers_runtime))
print("Number of votes outliers:", len(outliers_votes))

inconsistent_movies = df[df['revenue'] < df['budget']]
print("Number of inconsistent movies (revenue < budget):", len(inconsistent_movies))

# PART 2: DESCRIPTIVE STATISTICS 

numerical_cols = ['budget','revenue','runtime','imdb_rating','metascore','votes','oscar_nominations','oscar_wins']

# 2.1 Numerical summary
summary = df[numerical_cols].agg(['mean','median','var','std','min','max','count'])
summary.loc['range'] = summary.loc['max'] - summary.loc['min']
summary.loc['IQR'] = df[numerical_cols].quantile(0.75) - df[numerical_cols].quantile(0.25)
print("\nNumerical summary:\n", summary)

# 2.2 Categorical summary
categorical_cols = ['genre','director','main_actor']
for col in categorical_cols:
    print(f"Column: {col}")
    print("Unique values:", df[col].nunique())
    print("Top 10 most frequent:")
    print(df[col].value_counts().head(10))
    print("Proportions (%):")
    print(df[col].value_counts(normalize=True).head(10)*100)

# 2.3
cv = df[numerical_cols].std() / df[numerical_cols].mean()
print("Coefficient of Variation (CV):\n", cv)

skewness = 3 * (df[numerical_cols].mean() - df[numerical_cols].median()) / df[numerical_cols].std()
print("\nSkewness (3*(mean-median)/std):\n", skewness)

"""
2.3 (3)
1. Budgets and revenues vary greatly, with an average budget of about $36M and average revenue of $153M, while some movies generate extremely high revenues as outliers.
2. Genres are distributed almost evenly, with Sci-Fi at 13%, Action at 12.9%, and Thriller at 11.6%, showing no single dominant genre.
3. Some actors and directors appear frequently, for example, Ryan Gosling is in 11% of movies and Patty Jenkins directed 10%, indicating recurring “top” industry figures.
"""

# PART 3: DISTRIBUTION ANALYSIS 

# 3.1
fig, axes = plt.subplots(2,4, figsize=(20,10))
for i, col in enumerate(numerical_cols):
    ax_hist = axes[0,i%4]
    ax_box = axes[1,i%4]
    
    sns.histplot(df[col], kde=True, ax=ax_hist)
    ax_hist.axvline(df[col].mean(), color='red', linestyle='--')
    ax_hist.axvline(df[col].median(), color='green', linestyle='-')
    ax_hist.set_title(f'Histogram {col}')
    
    sns.boxplot(x=df[col], ax=ax_box)
    ax_box.set_title(f'Boxplot {col}')
plt.tight_layout()
plt.show()

# 3.2
columns = ['budget', 'revenue', 'votes']
outliers_list = []
for col in columns:
    outliers_list.append(find_outliers(col))

outliers_df = pd.concat(outliers_list).drop_duplicates()
total_revenue = df['revenue'].sum()
outliers_revenue = outliers_df['revenue'].sum()

percentage = (outliers_revenue / total_revenue) * 100

print(f"Outlier movies contribute {percentage:.2f}% of total revenue.")

top5 = outliers_df.sort_values(by='revenue', ascending=False).head(5)
print(top5[['movie_id', 'budget', 'revenue', 'votes']])

"""
Some movies have no many votes but can still be outliers in budget and revenue; the five selected outlier movies have high budgets, high revenue, showing their popularity and confirming that they are real blockbusters, not data errors.
"""

# 3.3
stats.probplot(df['imdb_rating'], dist="norm", plot=plt)
plt.title('Q-Q plot for IMDb Rating')
plt.show()

ratings = df['imdb_rating'].dropna()
mean = ratings.mean()
std = ratings.std()

within_1_std = ((ratings >= mean - std) & (ratings <= mean + std)).sum()
within_2_std = ((ratings >= mean - 2*std) & (ratings <= mean + 2*std)).sum()
within_3_std = ((ratings >= mean - 3*std) & (ratings <= mean + 3*std)).sum()

total = len(ratings)

print("1 std:", within_1_std, "(", within_1_std/total*100, "% )")
print("2 std:", within_2_std, "(", within_2_std/total*100, "% )")
print("3 std:", within_3_std, "(", within_3_std/total*100, "% )")

# PART 4: RELATIONSHIP ANALYSIS

# 4.1
numerical_cols = ['budget', 'revenue', 'runtime', 'imdb_rating', 'metascore', 'votes', 'oscar_nominations', 'oscar_wins']
corr_matrix = df[numerical_cols].corr()
print(corr_matrix)
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# 4.2 Scatter plot matrix
plt.figure(figsize=(12,10))
sns.pairplot(df, vars=['budget','revenue','imdb_rating','votes'], hue='genre', kind='scatter')
plt.show()

# 4.3 
df['profit_margin'] = np.where(df['budget']==0, np.nan, (df['revenue']-df['budget'])/df['budget'])
corr = df[['profit_margin','imdb_rating']].corr()
print("Correlation:", corr)
sns.scatterplot(x='profit_margin', y='imdb_rating', data=df)
plt.show()

# PART 5: COMPARATIVE ANALYSIS 

# 5.1 Genre comparison
plt.figure(figsize=(12,5))
sns.boxplot(x='genre', y='budget', data=df)
plt.title('Budget by Genre')
plt.show()

plt.figure(figsize=(12,5))
sns.boxplot(x='genre', y='revenue', data=df)
plt.title('Revenue by Genre')
plt.show()

plt.figure(figsize=(12,5))
sns.violinplot(x='genre', y='imdb_rating', data=df)
plt.title('IMDb Rating by Genre')
plt.show()

profit_by_genre = df.groupby('genre')['profit_margin'].mean().sort_values(ascending=False)
print("\nAverage profit margin by genre:\n", profit_by_genre)

oscar_by_genre = df.groupby('genre')['oscar_wins'].sum().sort_values(ascending=False)
print("\nTotal Oscar wins by genre:\n", oscar_by_genre)

rating_sd_by_genre = df.groupby('genre')['imdb_rating'].std()
print(rating_sd_by_genre)

# 5.2 
df['release_year'] = df['release_year'].astype(int)
budget_avg = df.groupby('release_year')['budget'].mean().rolling(3).mean()
budget_avg.plot(title='Average Budget by Year')
plt.show()

rating_avg = df.groupby('release_year')['imdb_rating'].mean()
rating_avg.plot(title='Average IMDb Rating by Year')
plt.show()
correlation = df[['release_year','imdb_rating']].corr().iloc[0,1]
print(f"Correlation between release_year and imdb_rating: {correlation:.2f}")

# 5.3
top_directors = df['director'].value_counts().head(10).index
top_df = df[df['director'].isin(top_directors)]
director_stats = top_df.groupby('director').agg({
    'imdb_rating':'mean',
    'profit_margin':'mean',
    'oscar_nominations':'mean'
}).sort_values('imdb_rating', ascending=False)
print("\nTop 10 directors stats:\n", director_stats)