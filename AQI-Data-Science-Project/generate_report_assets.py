import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import os

os.makedirs('outputs/charts', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

df = pd.read_csv('dataset/global_urban_aqi_dataset.csv')
df.drop_duplicates(inplace=True)
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

df.dropna(inplace=True)

def get_aqi_category(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Moderate'
    elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200: return 'Unhealthy'
    elif aqi <= 300: return 'Very Unhealthy'
    else: return 'Hazardous'
df['AQI Category'] = df['AQI'].apply(get_aqi_category)

# EDA
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='AQI Category', order=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
plt.savefig('outputs/charts/aqi_distribution.png')
plt.close()

plt.figure(figsize=(12, 6))
avg_aqi_country = df.groupby('Country')['AQI'].mean().sort_values(ascending=False)
sns.barplot(x=avg_aqi_country.index, y=avg_aqi_country.values)
plt.savefig('outputs/charts/avg_aqi_country.png')
plt.close()

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Year', y='AQI', errorbar=None, marker='o')
plt.savefig('outputs/charts/aqi_trend_year.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PM2.5', y='AQI', hue='AQI Category', alpha=0.7)
plt.savefig('outputs/charts/pm25_vs_aqi.png')
plt.close()

plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.savefig('outputs/charts/correlation_heatmap.png')
plt.close()

# Basic Stats
with open('outputs/results/stats.txt', 'w') as f:
    f.write(f"Mean AQI: {df['AQI'].mean()}\\n")
    f.write(f"Min AQI: {df['AQI'].min()}\\n")
    f.write(f"Max AQI: {df['AQI'].max()}\\n")
    f.write(f"Std AQI: {df['AQI'].std()}\\n")
    highest_aqi_loc = df.loc[df['AQI'].idxmax()]
    lowest_aqi_loc = df.loc[df['AQI'].idxmin()]
    f.write(f"Highest: {highest_aqi_loc['City']}, {highest_aqi_loc['Country']} ({highest_aqi_loc['AQI']})\\n")
    f.write(f"Lowest: {lowest_aqi_loc['City']}, {lowest_aqi_loc['Country']} ({lowest_aqi_loc['AQI']})\\n")

# ML
features = ['PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2', 'Temperature', 'Humidity', 'Wind Speed']
X = df[features]
y = df['AQI Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))

nb = GaussianNB()
nb.fit(X_train, y_train)
acc_nb = accuracy_score(y_test, nb.predict(X_test))

# K-Means
X_scaled_full = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled_full)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.8)
plt.savefig('outputs/charts/pca_clusters.png')
plt.close()

with open('outputs/results/ml_results.txt', 'w') as f:
    f.write(f"KNN Accuracy: {acc_knn}\\n")
    f.write(f"Naive Bayes Accuracy: {acc_nb}\\n")
    f.write(f"PCA Var: {sum(pca.explained_variance_ratio_)*100:.2f}%\\n")

print("Generated charts and results!")
