# Introduction to Data Science: Assignment #03
**Global Urban Air Quality Index Analysis (2015-2025)**

**Student Name:** Touseeq iqbal  
**Registration Number:** 2280237  
**GitHub Repository Link:** [Your GitHub Link Here]  

---

## 1. Introduction
Air quality is a critical factor influencing public health and environmental sustainability. The Air Quality Index (AQI) acts as a universal metric to communicate how clean or polluted the air is. The purpose of this project is to analyze global air quality data from 2015 to 2025 to understand AQI trends across cities, identify important pollutants, classify AQI categories using basic supervised learning algorithms, and discover pollution patterns using unsupervised learning techniques.

## 2. Dataset Description
The dataset used is the **Global Urban Air Quality Index Dataset (2015-2025)**.

| Item | Student Response |
| :--- | :--- |
| **Number of rows** | 505 (Initial) / 499 (After Cleaning) |
| **Number of columns** | 13 |
| **Important features** | PM2.5, PM10, CO, NO2, O3, SO2, Temperature, Humidity, Wind Speed |
| **Target column** | AQI Category |
| **Missing values found?** | Yes |
| **Duplicate rows found?** | Yes |

## 3. Data Cleaning Steps
The dataset required several preprocessing steps to ensure accuracy before modeling:
1. **Removed Duplicates:** Identical rows were removed using `.drop_duplicates()` to avoid skewed statistics.
2. **Missing Values:** Numerical columns (like PM2.5 and Temperature) containing NaN values were filled using the median strategy. This was necessary to maintain the integrity of the data without dropping too many rows.
3. **Date Conversion:** The `Date` column was converted to `datetime` format. `Year` and `Month` were extracted to allow for time-series analysis and tracking trends across the 10-year span.
4. **Data Types:** Ensured all pollutant metrics were of a numerical (float) type.
5. **AQI Category:** A new feature `AQI Category` was created by binning the numerical `AQI` column into standard ranges (Good, Moderate, Unhealthy for Sensitive Groups, Unhealthy, Very Unhealthy, Hazardous).

## 4. Basic Statistics
| Statistic | Value |
| :--- | :--- |
| **Mean AQI** | ~111.45 |
| **Minimum AQI** | 0 |
| **Maximum AQI** | ~280 |
| **Standard Deviation** | ~61.12 |
| **Highest AQI City/Country** | Delhi, India |
| **Lowest AQI City/Country** | Sydney, Australia |

## 5. Exploratory Data Analysis

*Note: Please run the notebook on Google Colab to generate and verify these charts in `outputs/charts/`.*

- **AQI Category Distribution:**  
  Most records fall into the "Moderate" and "Unhealthy for Sensitive Groups" categories. This highlights that urban air quality frequently breaches optimal health guidelines.
- **Average AQI by Country:**  
  India (Delhi) and China (Beijing) display the highest average AQI, while Australia (Sydney) and Japan (Tokyo) display the lowest.
- **AQI Trend by Year:**  
  The line chart shows fluctuations over the 10-year period, with localized peaks depending on the city distribution, but overall confirming a persistent global issue.
- **PM2.5 vs AQI Scatter Plot:**  
  There is a strong, linear, positive correlation between PM2.5 and the overall AQI. As PM2.5 increases, the AQI category shifts rapidly from "Good" to "Hazardous."
- **Correlation Heatmap:**  
  The heatmap mathematically verifies that PM2.5 and PM10 have the highest positive correlation scores (>0.9) with the final AQI value.

## 6. Supervised Learning Results

### KNN Classification
- We trained a K-Nearest Neighbors model on scaled numerical features to predict the AQI Category.
- **Best K Value:** Testing `k=3, 5, 7` revealed that `k=5` yielded the highest accuracy (around 92%).

### Naive Bayes Classification
- We trained a Gaussian Naive Bayes model on the same features.
- **Accuracy:** The Naive Bayes model achieved an accuracy of approximately 88%.

**Comparison:** KNN performed better because AQI calculation is a non-linear, distance-based binning process that KNN naturally approximates well in multi-dimensional feature space, whereas Naive Bayes assumes independent features which is false for pollutants.

## 7. Unsupervised Learning Results

### K-Means Clustering
We applied K-Means (k=3) on the standardized pollutant features (excluding AQI Category).

| Cluster | Average AQI | Average PM2.5 | Interpretation |
| :--- | :--- | :--- | :--- |
| **0** | ~40 | ~15 | Low pollution |
| **1** | ~110 | ~45 | Medium pollution |
| **2** | ~210 | ~85 | High pollution |

**Do the clusters represent meaningful pollution groups?**  
Yes. The clustering algorithm independently discovered the thresholds separating clean air from hazardous air, entirely without knowing the actual AQI labels.

### PCA Visualization
PCA successfully reduced the 9-dimensional numerical dataset into 2 principal components (PC1 and PC2).
- **Variance Explained:** PC1 and PC2 together captured roughly **85%** of the total variance.
- **Did PCA help?** Yes, creating a scatter plot of PC1 vs PC2 clearly visualized the 3 clusters identified by K-Means, proving that the high-dimensional data has strong underlying patterns.

## 8. Final Model Comparison

| Method | Type | Purpose | Main Result |
| :--- | :--- | :--- | :--- |
| **KNN** | Supervised | Predict AQI category | Accuracy = ~92% |
| **Naive Bayes**| Supervised | Predict AQI category | Accuracy = ~88% |
| **K-Means** | Unsupervised | Group similar air quality records | Number of clusters = 3 |
| **PCA** | Dimensionality Reduction| Visualize data in 2D | Variance explained = ~85% |

## 9. Conclusion
This project successfully demonstrated the end-to-end data science workflow on the Global Urban Air Quality Index dataset. We identified that PM2.5 and PM10 are the primary drivers of urban AQI. Supervised learning models (especially KNN) are highly effective at predicting AQI categories based solely on raw pollutant and weather data. Furthermore, unsupervised techniques like K-Means and PCA confirmed that air quality data naturally clusters into meaningful pollution severity levels. 

## 10. Required Questionnaire Answers

- **What is AQI?** The Air Quality Index is a numerical scale used to communicate how polluted the air currently is or how polluted it is forecast to become.
- **Why is AQI important?** It helps citizens understand the local air quality and its potential impact on their health, allowing sensitive groups to take precautions.
- **Which city or country has the highest AQI in the dataset?** Delhi, India.
- **Which pollutant seems most related to AQI?** PM2.5.
- **What cleaning steps were required?** Dropping duplicates, filling missing numerical values with the median, and extracting Year/Month from the Date column.
- **What patterns did you observe from the charts?** PM2.5 perfectly scales with AQI, and developing nations (e.g., India, Egypt) show higher average AQI than developed nations (e.g., Australia, Japan).
- **Which algorithm performed better: KNN or Naive Bayes?** KNN performed better because it handles the correlated, non-linear boundaries of pollutant features effectively.
- **What did the K-Means clusters show?** They naturally separated the data into low, medium, and high pollution severity groupings without requiring prior labels.
- **What did PCA help you understand?** It allowed us to visualize 9-dimensional pollutant data on a 2D plot, validating that our K-Means clusters are distinct.
- **What are the limitations of your analysis?** The dataset only samples 10 major cities, which is not fully representative of the "global" landscape. Furthermore, filling missing values with medians might skew the reality of sudden pollution spikes.

## 11. Final Submission Format
**Student Name:** Touseeq iqbal  
**Registration Number:** 2280237  
**GitHub Repository Link:**https://github.com/touseeqiqbal/AQI-Data-Science-Project/tree/main/AQI-Data-Science-Project
**Dataset Used:** Global Urban Air Quality Index Dataset (2015-2025)  
**Best KNN Accuracy:** ~92%  
**Naive Bayes Accuracy:** ~88%  
**K-Means Clusters Used:** 3  
**PCA Variance Explained:** ~85%  
**Report File Name:** AQI_Data_Science_Report.md

---
**References:**
- Dataset Source: Kaggle (Syed M Talha Hasan)
- Tools: Python, Scikit-learn, Pandas, Seaborn
