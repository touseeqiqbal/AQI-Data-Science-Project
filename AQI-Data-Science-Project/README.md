# AQI-Data-Science-Project

## Project Information
**Student Name:** Touseeq iqbal
**Registration Number:** 2280237
**Dataset Name:** Global Urban Air Quality Index Dataset (2015-2025)

## Problem Statement
This project analyzes global air quality data from 2015 to 2025. The purpose is to understand AQI trends across cities and countries, identify important pollutants, classify AQI categories using basic supervised learning algorithms (KNN and Naive Bayes), and discover pollution patterns using unsupervised learning techniques (K-Means and PCA).

## Tools and Libraries Used
- **Python**: Primary programming language
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical computations
- **Matplotlib & Seaborn**: For exploratory data visualization
- **Scikit-Learn**: For machine learning tasks (KNN, Naive Bayes, K-Means, PCA)
- **Jupyter Notebook / Google Colab**: Development environment

## How to Run the Notebook
1. **Clone or Download the Repository:** Extract the files to your local machine.
2. **Download the Dataset:** Download the `global-urban-air-quality-index-dataset-2015-2025.csv` from Kaggle and place it in the `dataset/` folder. Ensure it is named `global_urban_aqi_dataset.csv`.
3. **Open Google Colab:** Navigate to [Google Colab](https://colab.research.google.com/).
4. **Upload the Notebook:** Go to `File` -> `Upload notebook` and upload `AQI_Data_Science_Project.ipynb` from the `notebook/` folder.
5. **Upload the Dataset:** On the left sidebar in Colab, click the folder icon ("Files") and upload `global_urban_aqi_dataset.csv` into the session storage.
6. **Run All Cells:** Go to `Runtime` -> `Run all` to execute the full data science workflow.

## Summary of Main Findings
- **Data Distribution**: The majority of the observations fall into the "Moderate" and "Unhealthy for Sensitive Groups" categories.
- **Pollutants**: PM2.5 and PM10 showed the strongest positive correlation with the overall AQI.
- **Supervised Learning**: K-Nearest Neighbors (KNN) generally performed slightly better than Naive Bayes in classifying the AQI category accurately, capturing the non-linear boundaries.
- **Unsupervised Learning**: K-Means clustering effectively partitioned the dataset into 3 meaningful clusters: Low, Medium, and High pollution, clearly separating based on PM2.5 and AQI values. PCA visualization helped confirm these clusters in a 2D space, capturing the majority of the dataset's variance.

## Screenshots of Important Charts
(Placeholder for screenshots. After running the notebook, copy charts from `outputs/charts/` and embed them here, e.g., `![AQI Distribution](../outputs/charts/aqi_distribution.png)`)

## Report File Location
The final detailed written report is located at: `report/AQI_Data_Science_Report.pdf` (or `.md` format in the same directory).
