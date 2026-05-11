import json
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_dummy_data():
    os.makedirs('dataset', exist_ok=True)
    np.random.seed(42)
    
    num_rows = 500
    locations = [
        ('New York', 'USA'), ('London', 'UK'), ('Beijing', 'China'), 
        ('Delhi', 'India'), ('Tokyo', 'Japan'), ('Paris', 'France'),
        ('Sydney', 'Australia'), ('Cairo', 'Egypt'), ('Berlin', 'Germany'),
        ('Sao Paulo', 'Brazil')
    ]
    
    data = []
    start_date = datetime(2015, 1, 1)
    for _ in range(num_rows):
        city, country = random.choice(locations)
        days_offset = random.randint(0, 365 * 10)
        date = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        # Base AQI based on city to make it realistic
        base_aqi = {'Beijing': 150, 'Delhi': 200, 'New York': 50, 'London': 60, 'Tokyo': 40, 'Paris': 55, 'Sydney': 30, 'Cairo': 120, 'Berlin': 45, 'Sao Paulo': 80}[city]
        aqi = max(0, int(np.random.normal(base_aqi, 30)))
        
        # Correlated pollutants
        pm25 = max(0, aqi * 0.4 + np.random.normal(0, 5))
        pm10 = max(0, aqi * 0.6 + np.random.normal(0, 10))
        co = max(0, aqi * 0.05 + np.random.normal(0, 1))
        no2 = max(0, aqi * 0.1 + np.random.normal(0, 2))
        o3 = max(0, aqi * 0.2 + np.random.normal(0, 4))
        so2 = max(0, aqi * 0.05 + np.random.normal(0, 1))
        
        temp = np.random.normal(20, 10)
        humidity = np.random.normal(60, 20)
        wind_speed = max(0, np.random.normal(10, 5))
        
        data.append([city, country, date, pm25, pm10, co, no2, o3, so2, temp, humidity, wind_speed, aqi])
        
    df = pd.DataFrame(data, columns=['City', 'Country', 'Date', 'PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2', 'Temperature', 'Humidity', 'Wind Speed', 'AQI'])
    
    # Introduce some missing values and duplicates for cleaning requirements
    df.loc[10:15, 'PM2.5'] = np.nan
    df.loc[20:25, 'Temperature'] = np.nan
    df = pd.concat([df, df.iloc[0:5]], ignore_index=True) # duplicates
    
    df.to_csv('dataset/global_urban_aqi_dataset.csv', index=False)
    print("Generated dummy dataset at dataset/global_urban_aqi_dataset.csv")

def create_notebook():
    os.makedirs('notebook', exist_ok=True)
    os.makedirs('outputs/charts', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)
    
    cells = []
    
    def add_md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split('\n')]})
        
    def add_code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split('\n')]})

    add_md("# Global Urban Air Quality Index Dataset (2015-2025) Analysis\n**Introduction to Data Science - Assignment # 03**")
    
    add_md("## Part A: Data Loading and Understanding")
    add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Create output directories if they don't exist
os.makedirs('../outputs/charts', exist_ok=True)
os.makedirs('../outputs/results', exist_ok=True)

# Load the dataset
file_path = 'global_urban_aqi_dataset.csv' if os.path.exists('global_urban_aqi_dataset.csv') else '../dataset/global_urban_aqi_dataset.csv'
df = pd.read_csv(file_path)""")

    add_code("""# 2. Display first 5 rows
display(df.head())""")

    add_code("""# 3. Show number of rows and columns
print("Rows and Columns:", df.shape)""")

    add_code("""# 4. Display all column names
print("Column Names:")
print(df.columns.tolist())""")

    add_code("""# 5. Display data types
print("Data Types:")
print(df.dtypes)""")

    add_code("""# 6. Check missing values
print("Missing Values:")
print(df.isnull().sum())""")

    add_code("""# 7. Check duplicate records
print("Duplicate Records:", df.duplicated().sum())""")

    add_md("### Short Explanation of Important Columns\n- **City/Country**: The location of the air quality reading.\n- **Date**: When the reading was taken.\n- **PM2.5, PM10, CO, NO2, O3, SO2**: Various pollutants that contribute to air pollution.\n- **AQI**: The overall Air Quality Index, which is our target variable for prediction.")

    add_md("## Part B: Data Cleaning")
    add_code("""# 9. Remove duplicate rows
df.drop_duplicates(inplace=True)
print("Duplicates after removal:", df.duplicated().sum())""")

    add_code("""# 10. Handle missing values (Fill numerical with median)
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
print("Missing values after filling:")
print(df.isnull().sum())""")

    add_code("""# 11 & 12. Convert date and create year/month columns
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
display(df[['Date', 'Year', 'Month']].head())""")

    add_code("""# 13 & 14. Ensure numerical format (done above implicitly, but let's double check)
df.dropna(inplace=True) # Drop any remaining NaNs in categorical fields
print(df.shape)""")

    add_md("## Part C: AQI Category Creation")
    add_code("""# Create AQI categories based on standard ranges
def get_aqi_category(aqi):
    if aqi <= 50: return 'Good'
    elif aqi <= 100: return 'Moderate'
    elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200: return 'Unhealthy'
    elif aqi <= 300: return 'Very Unhealthy'
    else: return 'Hazardous'

if 'AQI Category' not in df.columns and 'AQI' in df.columns:
    df['AQI Category'] = df['AQI'].apply(get_aqi_category)

display(df[['AQI', 'AQI Category']].head())""")

    add_md("## Part D: Exploratory Data Analysis")
    add_code("""# 1. AQI category distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='AQI Category', order=['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
plt.title('AQI Category Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../outputs/charts/aqi_distribution.png')
plt.show()""")

    add_md("**Explanation:** This chart shows how many records fall into each AQI class. It helps us understand the general air quality distribution in our dataset.")

    add_code("""# 2. Average AQI by Country
plt.figure(figsize=(12, 6))
avg_aqi_country = df.groupby('Country')['AQI'].mean().sort_values(ascending=False)
sns.barplot(x=avg_aqi_country.index, y=avg_aqi_country.values)
plt.title('Average AQI by Country')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../outputs/charts/avg_aqi_country.png')
plt.show()""")

    add_md("**Explanation:** Comparing average AQI across countries reveals which regions suffer the most from air pollution on average.")

    add_code("""# 3. AQI trend by year
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Year', y='AQI', errorbar=None, marker='o')
plt.title('AQI Trend by Year (2015-2025)')
plt.tight_layout()
plt.savefig('../outputs/charts/aqi_trend_year.png')
plt.show()""")

    add_md("**Explanation:** The line chart illustrates the overall global trend of air quality over the 10-year period, indicating if pollution is worsening or improving.")

    add_code("""# 4. PM2.5 vs AQI scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PM2.5', y='AQI', hue='AQI Category', alpha=0.7)
plt.title('PM2.5 vs AQI')
plt.tight_layout()
plt.savefig('../outputs/charts/pm25_vs_aqi.png')
plt.show()""")

    add_md("**Explanation:** This scatter plot highlights the strong relationship between PM2.5 concentrations and the overall AQI value.")

    add_code("""# 5. Correlation heatmap
plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.savefig('../outputs/charts/correlation_heatmap.png')
plt.show()""")

    add_md("**Explanation:** The heatmap shows linear relationships between all numerical variables. We can clearly see which pollutants are most correlated with AQI.")

    add_md("## Basic Statistics Requirement")
    add_code("""print("Mean AQI:", df['AQI'].mean())
print("Minimum AQI:", df['AQI'].min())
print("Maximum AQI:", df['AQI'].max())
print("Standard deviation of AQI:", df['AQI'].std())

highest_aqi_loc = df.loc[df['AQI'].idxmax()]
lowest_aqi_loc = df.loc[df['AQI'].idxmin()]

print(f"Highest AQI Location: {highest_aqi_loc['City']}, {highest_aqi_loc['Country']} (AQI: {highest_aqi_loc['AQI']})")
print(f"Lowest AQI Location: {lowest_aqi_loc['City']}, {lowest_aqi_loc['Country']} (AQI: {lowest_aqi_loc['AQI']})")""")

    add_md("## Supervised Learning Task\n### Part E: KNN Classification")
    add_code("""# Select numerical features and target
features = ['PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2', 'Temperature', 'Humidity', 'Wind Speed']
X = df[features]
y = df['AQI Category']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test multiple values of k
for k in [3, 5, 7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"KNN Accuracy (k={k}): {acc:.4f}")

# Detailed report for best K (let's say k=5)
knn_best = KNeighborsClassifier(n_neighbors=5)
knn_best.fit(X_train_scaled, y_train)
y_pred_knn = knn_best.predict(X_test_scaled)

print("\\nConfusion Matrix (KNN k=5):\\n", confusion_matrix(y_test, y_pred_knn))
print("\\nClassification Report (KNN k=5):\\n", classification_report(y_test, y_pred_knn, zero_division=0))""")

    add_md("### Part F: Naive Bayes Classification")
    add_code("""# Train Gaussian Naive Bayes
nb = GaussianNB()
# Naive bayes can work with unscaled data, but scaled is fine too. Let's use unscaled for traditional approach
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print("\\nConfusion Matrix (Naive Bayes):\\n", confusion_matrix(y_test, y_pred_nb))
print("\\nClassification Report (Naive Bayes):\\n", classification_report(y_test, y_pred_nb, zero_division=0))""")

    add_md("## Unsupervised Learning Task\n### Part G: K-Means Clustering")
    add_code("""# Remove AQI Category (already done, X contains only numerical features)
# Standardize the data (using all X now)
X_scaled_full = scaler.fit_transform(X)

# Apply K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled_full)

# Table showing average values per cluster
cluster_summary = df.groupby('Cluster')[features + ['AQI']].mean()
display(cluster_summary)""")

    add_md("### Part H: PCA Visualization")
    add_code("""# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled_full)

df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

# Plot PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='viridis', alpha=0.8)
plt.title('PCA Visualization of Pollution Clusters')
plt.tight_layout()
plt.savefig('../outputs/charts/pca_clusters.png')
plt.show()

# Explained variance
print(f"Explained Variance by PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"Explained Variance by PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_)*100:.2f}%")""")

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('notebook/AQI_Data_Science_Project.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    print("Generated Jupyter Notebook at notebook/AQI_Data_Science_Project.ipynb")

if __name__ == "__main__":
    generate_dummy_data()
    create_notebook()
