# NTI_Final_Project
# 📊 Sales Data Analysis & Profit Prediction App

An end-to-end data analytics and machine learning project that ingests raw sales data, performs rigorous data cleaning, conducts exploratory data analysis (EDA), and deploys a Random Forest predictive model to forecast transaction profitability. The project features comprehensive interactive dashboards and a Streamlit web application.

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Predictive Modeling](#predictive-modeling)
- [Streamlit Application](#streamlit-application)
- [Dashboards](#dashboards)
- [Team Members & Acknowledgements](#team-members--acknowledgements)
- [Installation & Usage](#installation--usage)

## 🎯 Project Overview
This project analyzes a retail sales dataset containing `Orders`, `Return`, and `People` records. The main goal is to uncover business insights regarding revenue, profitability, and customer segments, and to build a robust machine learning pipeline capable of predicting profit based on transaction features. 

**Overall KPIs achieved:**
* Total Orders: 5,006
* Total Sales: $2.30M
* Total Profit: $287K
* Profit Margin: 12.5%
* Return Rate: 6%

## 🧹 Data Cleaning & Preprocessing
The initial dataset contained 9,993 rows and 47 columns, many of which were spreadsheet artifacts. 
* **Dimensionality Reduction:** Dropped completely empty columns (e.g., `Column21` to `Column32`, `Unnamed: 40` to `Unnamed: 46`).
* **Missing Values & Duplicates:** Dropped duplicate segments (`Segment2`) and removed 15 rows with null values to achieve a clean 9,979 row dataset.
* **Standardization:** Stripped whitespaces, capitalized string values, cast `Quantity` to integer, and formatted `Postal Code` to a 5-character string (zero-padded).
* **Merging:** Joined the `Orders` dataset with `Returns` (binary mapping) and `People` (Regional Managers).

## 📈 Exploratory Data Analysis (EDA)
Comprehensive EDA was performed to uncover sales and profitability trends:
* **Category Performance:** Technology leads in both sales (36%) and profit, while Furniture yields high sales but very low profit margins.
* **Geographic Trends:** California, New York, and Washington are the most profitable states. Texas, Ohio, and Pennsylvania suffer from the lowest (negative) profits, heavily impacted by discounts.
* **Temporal Trends:** Sales show a steady year-over-year increase from 2014 to 2017.

## 🤖 Predictive Modeling
* **Target Variable:** `Profit`
* **Features Used:** 
  * *Numerical:* Sales, Discount, Quantity, Cost Of Items
  * *Categorical:* Category, Sub-Category, Segment
* **Model:** Random Forest Regressor (80/20 Train-Test Split) with a Scikit-Learn `Pipeline` and `ColumnTransformer` (OneHotEncoding).
* **Performance:** 
  * **MAE:** 8.20
  * **RMSE:** 39.07
  * **R² Score:** 92.79% (Excellent variance explanation).
* **Feature Importance:** Sales (52.0%) and Cost Of Items (30.9%) are the strongest predictors of profit. 

## 🚀 Streamlit Application
The project includes a deployed Streamlit app where users can dynamically upload datasets and models to view predictions and EDA. 
* **Interactive Uploads:** Users can upload the `profit_model.joblib` and `Sales Dataset cleaned` Excel file via the sidebar.
* **Dynamic EDA:** View Summary Statistics, Top/Bottom States, Correlation Heatmaps, and Segment breakdowns.
* **Feature Insights:** Visualizes the model's feature importance extraction dynamically.

## 📊 Dashboards
Multiple dashboards were created providing granular views:
1. **Overview Dashboard:** High-level KPIs, YoY Sales, and Category distribution.
2. **Sales Dashboard:** Sales and Cost metrics by Segment, Category, and Manager.
3. **Geography Dashboard:** State hierarchy mappings and Discount Impact analysis.
4. **Products Dashboard:** Sub-category deep dives (Copiers/Phones driving profit, Tables/Bookcases causing losses).

## 👥 Team Members & Acknowledgements
**Instructor:** Hagar Hisham

**Team Members:**

* **Ahmed Ali** – Data Cleaning & Visualization
* **Mariem Mohamed Emam** – Modeling & Streamlit Implementation
* **Youssef Mohamed Elhossien** – Data Cleaning, EDA & Visualization
* **Mohamed Khalid Kamal Fahiem** *(Team Leader)* – Data Cleaning, Visualization & Modeling
---
*Developed with Python, Pandas, Scikit-Learn, and Streamlit.*
# 📊 Sales Data Analysis & Profit Prediction App

An end-to-end data analytics and machine learning project that ingests raw sales data, performs rigorous data cleaning, conducts exploratory data analysis (EDA), and deploys a Random Forest predictive model to forecast transaction profitability. The project features comprehensive interactive dashboards and a Streamlit web application.

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Predictive Modeling](#predictive-modeling)
- [Streamlit Application](#streamlit-application)
- [Dashboards](#dashboards)
- [Team Members & Acknowledgements](#team-members--acknowledgements)
- [Installation & Usage](#installation--usage)

## 🎯 Project Overview
This project analyzes a retail sales dataset containing `Orders`, `Return`, and `People` records. The main goal is to uncover business insights regarding revenue, profitability, and customer segments, and to build a robust machine learning pipeline capable of predicting profit based on transaction features. 

**Overall KPIs achieved:**
* Total Orders: 5,006
* Total Sales: $2.30M
* Total Profit: $287K
* Profit Margin: 12.5%
* Return Rate: 6%

## 🧹 Data Cleaning & Preprocessing
The initial dataset contained 9,993 rows and 47 columns, many of which were spreadsheet artifacts. 
* **Dimensionality Reduction:** Dropped completely empty columns (e.g., `Column21` to `Column32`, `Unnamed: 40` to `Unnamed: 46`).
* **Missing Values & Duplicates:** Dropped duplicate segments (`Segment2`) and removed 15 rows with null values to achieve a clean 9,979 row dataset.
* **Standardization:** Stripped whitespaces, capitalized string values, cast `Quantity` to integer, and formatted `Postal Code` to a 5-character string (zero-padded).
* **Merging:** Joined the `Orders` dataset with `Returns` (binary mapping) and `People` (Regional Managers).

## 📈 Exploratory Data Analysis (EDA)
Comprehensive EDA was performed to uncover sales and profitability trends:
* **Category Performance:** Technology leads in both sales (36%) and profit, while Furniture yields high sales but very low profit margins.
* **Geographic Trends:** California, New York, and Washington are the most profitable states. Texas, Ohio, and Pennsylvania suffer from the lowest (negative) profits, heavily impacted by discounts.
* **Temporal Trends:** Sales show a steady year-over-year increase from 2014 to 2017.

## 🤖 Predictive Modeling
* **Target Variable:** `Profit`
* **Features Used:** 
  * *Numerical:* Sales, Discount, Quantity, Cost Of Items
  * *Categorical:* Category, Sub-Category, Segment
* **Model:** Random Forest Regressor (80/20 Train-Test Split) with a Scikit-Learn `Pipeline` and `ColumnTransformer` (OneHotEncoding).
* **Performance:** 
  * **MAE:** 8.20
  * **RMSE:** 39.07
  * **R² Score:** 92.79% (Excellent variance explanation).
* **Feature Importance:** Sales (52.0%) and Cost Of Items (30.9%) are the strongest predictors of profit. 

## 🚀 Streamlit Application
The project includes a deployed Streamlit app where users can dynamically upload datasets and models to view predictions and EDA. 
* **Interactive Uploads:** Users can upload the `profit_model.joblib` and `Sales Dataset cleaned` Excel file via the sidebar.
* **Dynamic EDA:** View Summary Statistics, Top/Bottom States, Correlation Heatmaps, and Segment breakdowns.
* **Feature Insights:** Visualizes the model's feature importance extraction dynamically.

## 📊 Dashboards
Multiple dashboards were created providing granular views:
1. **Overview Dashboard:** High-level KPIs, YoY Sales, and Category distribution.
2. **Sales Dashboard:** Sales and Cost metrics by Segment, Category, and Manager.
3. **Geography Dashboard:** State hierarchy mappings and Discount Impact analysis.
4. **Products Dashboard:** Sub-category deep dives (Copiers/Phones driving profit, Tables/Bookcases causing losses).

## 👥 Team Members & Acknowledgements
**Instructor:** Hagar Hisham

**Team Members:**

* **Ahmed Ali** – Data Cleaning & Visualization
* **Mariem Mohamed Emam** – Modeling & Streamlit Implementation
* **Youssef Mohamed Elhossien** – Data Cleaning, EDA & Visualization
* **Mohamed Khalid Kamal Fahiem** *(Team Leader)* – Data Cleaning, Visualization & Modeling
---
*Developed with Python, Pandas, Matplotlib, Seaborn, Plotly, Scikit-Learn, Streamlit and Power Bi *
