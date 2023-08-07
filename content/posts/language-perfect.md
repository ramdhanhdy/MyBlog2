---
title: More Traffic, More Mystery? A Web Visit Paradox
date: 2019-08-07
description: "An Application of Johnson Neuman Interval Analysis on Marketing Campaign Data"
image: images/cctv2.jpeg
imageAltAttribute: CCTV Camera
tags:
   - writing 
   - lorem 
---


## Executive Summary

In the evolving e-commerce landscape, the understanding of customer behavior has been recognized as paramount. A comprehensive analysis of a marketing campaign data has been conducted to offer insights into the factors influencing customer responses to campaigns. The emphasis has been placed on variables such as spending habits, purchase recency, and website visit frequency. The findings have revealed:

1. A positive relationship between the frequency of web visits and a customer's response was observed. However, this relationship has been nuanced by the total spending of the customer, emphasizing the necessity to approach customer behavior with a holistic perspective.
2. The diminishing effect of total spending with increased web visits suggests a complex interaction between engagement and expenditure.
3. A model, deemed reliable through metrics such as AIC, PCP, and Tjur's R2, has been used to validate the insights.

From a strategic viewpoint, businesses are advised not only to drive web traffic but also to ensure that these visits lead to meaningful engagements and purchases.



## 1. Introduction

The advent of the digital age has ushered in a renewed understanding of consumer behavior. As a significant majority of transactions have transitioned online, the need to discern what drives a customer to respond to a campaign or finalize a purchase has become paramount.

In this report, a detailed exploration into the intricacies of customer responses has been undertaken. Insights have been drawn from a comprehensive dataset that encapsulates various facets of customer behavior. Through the exploratory data analysis, influential factors have been highlighted, new perspectives on traditionally overlooked variables have been presented, and a nuanced relationship between website engagement and spending has been showcased.

As the report is navigated, the complexities of customer behavior, the interplay between various influential factors, and recommendations on how these insights can be harnessed by businesses for optimized marketing campaigns will be revealed.

## 2. Technical Setup

### Python


```python
#data processing
import numpy as np
import pandas as pd
import os
import datetime
import math
import functools
from datetime import datetime
import scipy.stats as stats
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency


import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import sys
sys.path.append(r"D:\Project\rpy2_testing\src")
import my_utils

import warnings
warnings.filterwarnings('ignore')
```

### R


```python
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.1'  
```


```python
# import rpy2's package module
from rpy2.robjects.packages import importr
from functools import partial
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.ipython import html
html.html_rdataframe=partial(html.html_rdataframe, table_class="docutils")
```


```python
#RUn to install R packages if not installed yet
utils = importr('utils')
base = importr('base')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('peformance', 'tidyr')


# Selectively install what needs to be install.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
```


```python
%load_ext rpy2.ipython
```


```r
%%R
library(dplyr)
library(ggplot2)
library(ggstatsplot)
library(gridExtra)
library(kableExtra)
library(ggthemr)
library(summarytools)
```

## 3. Data Collection, Understanding, and Preparation

### 3.1. Data Collection


```python
df_raw = pd.read_csv('data\marketing_campaign.csv', sep='\t')
```

### 3.2. Data Understanding


```python
my_utils.dataframe_info(df_raw)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Data Type</th>
      <th>Unique Count</th>
      <th>Unique Sample</th>
      <th>Missing Values</th>
      <th>Missing Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ID</td>
      <td>int64</td>
      <td>2240</td>
      <td>[5524, 2174, 4141, 6182, 5324]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Year_Birth</td>
      <td>int64</td>
      <td>59</td>
      <td>[1957, 1954, 1965, 1984, 1981]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Education</td>
      <td>object</td>
      <td>5</td>
      <td>[Graduation, PhD, Master, Basic, 2n Cycle]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Marital_Status</td>
      <td>object</td>
      <td>8</td>
      <td>[Single, Together, Married, Divorced, Widow]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Income</td>
      <td>float64</td>
      <td>1974</td>
      <td>[58138.0, 46344.0, 71613.0, 26646.0, 58293.0]</td>
      <td>24</td>
      <td>1.0714</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Kidhome</td>
      <td>int64</td>
      <td>3</td>
      <td>[0, 1, 2]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Teenhome</td>
      <td>int64</td>
      <td>3</td>
      <td>[0, 1, 2]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Dt_Customer</td>
      <td>object</td>
      <td>663</td>
      <td>[04-09-2012, 08-03-2014, 21-08-2013, 10-02-201...</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Recency</td>
      <td>int64</td>
      <td>100</td>
      <td>[58, 38, 26, 94, 16]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MntWines</td>
      <td>int64</td>
      <td>776</td>
      <td>[635, 11, 426, 173, 520]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MntFruits</td>
      <td>int64</td>
      <td>158</td>
      <td>[88, 1, 49, 4, 43]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MntMeatProducts</td>
      <td>int64</td>
      <td>558</td>
      <td>[546, 6, 127, 20, 118]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MntFishProducts</td>
      <td>int64</td>
      <td>182</td>
      <td>[172, 2, 111, 10, 46]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MntSweetProducts</td>
      <td>int64</td>
      <td>177</td>
      <td>[88, 1, 21, 3, 27]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MntGoldProds</td>
      <td>int64</td>
      <td>213</td>
      <td>[88, 6, 42, 5, 15]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NumDealsPurchases</td>
      <td>int64</td>
      <td>15</td>
      <td>[3, 2, 1, 5, 4]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NumWebPurchases</td>
      <td>int64</td>
      <td>15</td>
      <td>[8, 1, 2, 5, 6]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NumCatalogPurchases</td>
      <td>int64</td>
      <td>14</td>
      <td>[10, 1, 2, 0, 3]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NumStorePurchases</td>
      <td>int64</td>
      <td>14</td>
      <td>[4, 2, 10, 6, 7]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NumWebVisitsMonth</td>
      <td>int64</td>
      <td>16</td>
      <td>[7, 5, 4, 6, 8]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>AcceptedCmp3</td>
      <td>int64</td>
      <td>2</td>
      <td>[0, 1]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>AcceptedCmp4</td>
      <td>int64</td>
      <td>2</td>
      <td>[0, 1]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>AcceptedCmp5</td>
      <td>int64</td>
      <td>2</td>
      <td>[0, 1]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>AcceptedCmp1</td>
      <td>int64</td>
      <td>2</td>
      <td>[0, 1]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>AcceptedCmp2</td>
      <td>int64</td>
      <td>2</td>
      <td>[0, 1]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Complain</td>
      <td>int64</td>
      <td>2</td>
      <td>[0, 1]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Z_CostContact</td>
      <td>int64</td>
      <td>1</td>
      <td>[3]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Z_Revenue</td>
      <td>int64</td>
      <td>1</td>
      <td>[11]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Response</td>
      <td>int64</td>
      <td>2</td>
      <td>[1, 0]</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



This dataset contains 29 variables, consisting of integers (int64), floating points (float64), and objects (object). 

- **ID**: A unique identifier for each customer. This feature may not contribute to a predictive model.

- **Year_Birth**: The birth year of customers, which can be converted to 'Age'.

- **Education** and **Marital_Status**: Categorical features that may require one-hot encoding.

- **Income**: A continuous feature indicating customer's income, with some missing values that need to be filled.

- **Kidhome** and **Teenhome**: Integer variables indicating the number of children in the household.

- **Dt_Customer**: The joining date of customers, which may need to be converted to 'tenure' or similar measure.

- **Recency**: Indicates how recently a customer made a purchase. Further context is needed for proper interpretation.

- **MntX** and **NumXPurchases**: Represent the amount spent on different categories of products and the number of purchases made through different channels, respectively.

- **AcceptedCmpX**: Binary features indicating whether the customer accepted offers in different campaigns.

- **Complain**: A binary feature indicating if the customer made a complaint recently.

- **Z_CostContact** and **Z_Revenue**: Features with no variance, hence can be removed.

- **Response**: The binary target variable indicating whether a customer accepted the latest offer.



### 3.2. Data Preparation

This phase of the data analysis project focuses on enhancing the quality of the dataset for further stages, which include exploratory data analysis, model building, and interpretation. 

Notes on processing that need to be done are as follows:

1. **Removal of Unnecessary Features**: Features such as 'ID', 'Z_CostContact', and 'Z_Revenue', which do not contribute significant insights for further analysis, are eliminated. 'ID' is merely a distinct identifier, while 'Z_CostContact' and 'Z_Revenue' do not display variability, making them uninformative for a predictive model.

2. **Handling Missing Values**: The 'Income' feature contains missing values, which require attention. Depending on the shape of the distribution, these missing values will be addressed using either its mean or median imputation. 

3. **Derivation of New Features**: New features, 'Age', 'Customer_Tenure', and 'Generation', are engineered from existing ones. 'Age' is calculated by subtracting the 'Year_Birth' from the current year. 'Customer_Tenure' reflects the customer's length of association with the store and is computed from the 'Dt_Customer' feature. The 'Generation' is determined based on the 'Age' feature and the definitions provided by Pew Research Centre. `Total_Purchases` and `Total_Spending` are calculated from `NumXPurchase` and `MntX` respectively.

4. **Encoding of Categorical Variables**: Categorical variables, namely 'Education' and 'Marital_Status', are encoded into a format suitable for machine learning algorithms. One-hot encoding is a suitable method for this purpose.

5. **Scaling of Numeric Features**: The features 'Income', 'Recency', 'MntX', and 'NumXPurchases' exist in different scales. To ensure that no single feature dominates the model. Scaling methods will be implemented later on as pipelines during experimentation since some machine learning models such as XGBoost and Catboost don't require feature scaling. This will be tested during the experimentation later on.

6. **Detection and Treatment of Outliers**: Outliers in features such as 'Income', 'Recency', 'MntX', and 'NumXPurchases' can significantly bias the model. Therefore, outlier detection and subsequent treatment, through methods like Z-score or the Interquartile Range (IQR) method, are employed.

**footnotes**:
> The 'Generation' feature is classified as per the definitions given by the Pew Research Centre:
>
> - The Silent Generation: Born 1928-1945 (76-93 years old)
> - Baby Boomers: Born 1946-1964 (57-75 years old)
> - Generation X: Born 1965-1980 (41-56 years old)
> - Millennials: Born 1981-1996 (25-40 years old)
> - Generation Z: Born 1997-2012 (9-24 years old)
> - Generation Alpha: Born 2010-2025 (0-11 years old)

#### 3.2.1 Unnecesary Features Removal, Missing Value Handling, and New Features Generation


```python
df = df_raw.copy(deep=True)

# 1. Redundant Features Removal
df = df.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1)

# 2. Missing Values Handling
df['Income'] = df['Income'].fillna(df['Income'].median())

# 3. New Features Derivation
# Calculate Age from Year_Birth
current_year = datetime.now().year
df['Age'] = current_year - df['Year_Birth']
df = df.drop('Year_Birth', axis=1)

# Calculate Total Purchase
df['Total_Purchases'] = df['NumDealsPurchases'] + df['NumWebPurchases'] + df['NumCatalogPurchases'] + df['NumStorePurchases']

# Calculate Customer_Tenure
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
last_recorded_date = df['Dt_Customer'].max()
df['Customer_Tenure'] = (last_recorded_date - df['Dt_Customer']).dt.days

# Calculate Total Spending
product_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[product_columns].sum(axis=1)

# Create Living_With feature based on Kidhome and Teenhome
conditions = [
    (df['Kidhome'] == 1) & (df['Teenhome'] == 0),
    (df['Kidhome'] == 0) & (df['Teenhome'] == 1),
    (df['Kidhome'] == 1) & (df['Teenhome'] == 1),
]
choices = ['With_Kids', 'With_Teens', 'With_Kids_and_Teens']
df['Living_With'] = np.select(conditions, choices, default='Alone')

# Create Is_Parent feature
df['Is_Parent'] = (df['Kidhome'] + df['Teenhome']).apply(lambda x: 1 if x > 0 else 0)

# Define generations
conditions = [
    (df['Age'] >= 76) & (df['Age'] <= 93),
    (df['Age'] >= 57) & (df['Age'] <= 75),
    (df['Age'] >= 41) & (df['Age'] <= 56),
    (df['Age'] >= 25) & (df['Age'] <= 40),
    (df['Age'] >= 9) & (df['Age'] <= 24),
    (df['Age'] >= 0) & (df['Age'] <= 11),
]
choices = ['Silent_Generation', 'Baby_Boomers', 'Generation_X', 'Millennials', 'Generation_Z', 'Generation_Alpha']
df['Generation'] = np.select(conditions, choices, default='Unknown')
```

### 3.2.2. Outlier Detection


```python
num_df = df.select_dtypes(include=np.number)
nunique_df = pd.DataFrame(data=num_df.apply(pd.Series.nunique), columns=['nunique']).rename_axis('variables')
selection = nunique_df[nunique_df['nunique'] > 5].index

#plotly frame setup
to_plot = num_df[selection]

# number of variables to plot
num_vars = len(to_plot.columns)

# calculate number of rows and columns for the subplot grid
plot_nrows = int(np.ceil(num_vars / 4))

# create subplot titles
subplot_titles = list(to_plot.columns) + [''] * (plot_nrows * 4 - num_vars) 

# create subplots
fig = make_subplots(rows=plot_nrows, cols=4, subplot_titles=tuple(subplot_titles))

# add traces
for i, col_name in enumerate(to_plot.columns):
    row = i // 4 + 1
    col = i % 4 + 1
    fig.add_trace(go.Box(y=to_plot[col_name], name=col_name), row=row, col=col)

fig.update_layout(height=1400, width=1200, showlegend=False, template='plotly_dark', title='Boxplot of Numerical Features')
fig.update_xaxes(visible=False, showticklabels=False)

fig.show()

```



<img src="figures/boxplots.png" width="90%" /> 

**Highlights**

The boxplot above shows that tere are possible outliers on the following features:
- Income, MntFruits, MntMeatProducts, MntSweetProducts, MntGoldProds, MntFishProducts, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumWebVisitsMonth, Age, and TotalSpent

Before deciding wether outliers found should be remove, let's calculate how many percentage of the data are considered as outliers by the Inter-quartile Range method used by the boxplot


```python
#count outliers
Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3-Q1


#outlier dataframe
outlier_count = ((num_df < (Q1-1.5*IQR)) | (num_df > (Q3 + 1.5*IQR))).sum() #outlier count 
outlier_df = pd.DataFrame(outlier_count, columns=['outlier_count']).rename_axis('variable')
outlier_df['percentage'] = outlier_df['outlier_count']/len(num_df)*100 #outlier percentage columns


pd.set_option('display.precision', 2)
outlier_df.sort_values('outlier_count', ascending=False).style.background_gradient(cmap='gist_heat').set_caption('Outlier Count')
```




<style type="text/css">
#T_c4159_row0_col0, #T_c4159_row0_col1 {
  background-color: #ffffff;
  color: #000000;
}
#T_c4159_row1_col0, #T_c4159_row1_col1 {
  background-color: #ff7d00;
  color: #f1f1f1;
}
#T_c4159_row2_col0, #T_c4159_row2_col1 {
  background-color: #ff5b00;
  color: #f1f1f1;
}
#T_c4159_row3_col0, #T_c4159_row3_col1 {
  background-color: #ff5500;
  color: #f1f1f1;
}
#T_c4159_row4_col0, #T_c4159_row4_col1 {
  background-color: #ed3d00;
  color: #f1f1f1;
}
#T_c4159_row5_col0, #T_c4159_row5_col1 {
  background-color: #c90d00;
  color: #f1f1f1;
}
#T_c4159_row6_col0, #T_c4159_row6_col1 {
  background-color: #c00100;
  color: #f1f1f1;
}
#T_c4159_row7_col0, #T_c4159_row7_col1, #T_c4159_row8_col0, #T_c4159_row8_col1 {
  background-color: #ba0000;
  color: #f1f1f1;
}
#T_c4159_row9_col0, #T_c4159_row9_col1 {
  background-color: #a50000;
  color: #f1f1f1;
}
#T_c4159_row10_col0, #T_c4159_row10_col1 {
  background-color: #620000;
  color: #f1f1f1;
}
#T_c4159_row11_col0, #T_c4159_row11_col1 {
  background-color: #270000;
  color: #f1f1f1;
}
#T_c4159_row12_col0, #T_c4159_row12_col1 {
  background-color: #210000;
  color: #f1f1f1;
}
#T_c4159_row13_col0, #T_c4159_row13_col1 {
  background-color: #1a0000;
  color: #f1f1f1;
}
#T_c4159_row14_col0, #T_c4159_row14_col1 {
  background-color: #180000;
  color: #f1f1f1;
}
#T_c4159_row15_col0, #T_c4159_row15_col1, #T_c4159_row16_col0, #T_c4159_row16_col1 {
  background-color: #090000;
  color: #f1f1f1;
}
#T_c4159_row17_col0, #T_c4159_row17_col1 {
  background-color: #040000;
  color: #f1f1f1;
}
#T_c4159_row18_col0, #T_c4159_row18_col1, #T_c4159_row19_col0, #T_c4159_row19_col1 {
  background-color: #030000;
  color: #f1f1f1;
}
#T_c4159_row20_col0, #T_c4159_row20_col1 {
  background-color: #020000;
  color: #f1f1f1;
}
#T_c4159_row21_col0, #T_c4159_row21_col1, #T_c4159_row22_col0, #T_c4159_row22_col1, #T_c4159_row23_col0, #T_c4159_row23_col1, #T_c4159_row24_col0, #T_c4159_row24_col1, #T_c4159_row25_col0, #T_c4159_row25_col1, #T_c4159_row26_col0, #T_c4159_row26_col1 {
  background-color: #000000;
  color: #f1f1f1;
}
</style>
<table id="T_c4159">
  <caption>Outlier Count</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_c4159_level0_col0" class="col_heading level0 col0" >outlier_count</th>
      <th id="T_c4159_level0_col1" class="col_heading level0 col1" >percentage</th>
    </tr>
    <tr>
      <th class="index_name level0" >variable</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_c4159_level0_row0" class="row_heading level0 row0" >Response</th>
      <td id="T_c4159_row0_col0" class="data row0 col0" >334</td>
      <td id="T_c4159_row0_col1" class="data row0 col1" >14.910714</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row1" class="row_heading level0 row1" >MntSweetProducts</th>
      <td id="T_c4159_row1_col0" class="data row1 col0" >248</td>
      <td id="T_c4159_row1_col1" class="data row1 col1" >11.071429</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row2" class="row_heading level0 row2" >MntFruits</th>
      <td id="T_c4159_row2_col0" class="data row2 col0" >227</td>
      <td id="T_c4159_row2_col1" class="data row2 col1" >10.133929</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row3" class="row_heading level0 row3" >MntFishProducts</th>
      <td id="T_c4159_row3_col0" class="data row3 col0" >223</td>
      <td id="T_c4159_row3_col1" class="data row3 col1" >9.955357</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row4" class="row_heading level0 row4" >MntGoldProds</th>
      <td id="T_c4159_row4_col0" class="data row4 col0" >207</td>
      <td id="T_c4159_row4_col1" class="data row4 col1" >9.241071</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row5" class="row_heading level0 row5" >MntMeatProducts</th>
      <td id="T_c4159_row5_col0" class="data row5 col0" >175</td>
      <td id="T_c4159_row5_col1" class="data row5 col1" >7.812500</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row6" class="row_heading level0 row6" >AcceptedCmp4</th>
      <td id="T_c4159_row6_col0" class="data row6 col0" >167</td>
      <td id="T_c4159_row6_col1" class="data row6 col1" >7.455357</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row7" class="row_heading level0 row7" >AcceptedCmp3</th>
      <td id="T_c4159_row7_col0" class="data row7 col0" >163</td>
      <td id="T_c4159_row7_col1" class="data row7 col1" >7.276786</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row8" class="row_heading level0 row8" >AcceptedCmp5</th>
      <td id="T_c4159_row8_col0" class="data row8 col0" >163</td>
      <td id="T_c4159_row8_col1" class="data row8 col1" >7.276786</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row9" class="row_heading level0 row9" >AcceptedCmp1</th>
      <td id="T_c4159_row9_col0" class="data row9 col0" >144</td>
      <td id="T_c4159_row9_col1" class="data row9 col1" >6.428571</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row10" class="row_heading level0 row10" >NumDealsPurchases</th>
      <td id="T_c4159_row10_col0" class="data row10 col0" >86</td>
      <td id="T_c4159_row10_col1" class="data row10 col1" >3.839286</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row11" class="row_heading level0 row11" >MntWines</th>
      <td id="T_c4159_row11_col0" class="data row11 col0" >35</td>
      <td id="T_c4159_row11_col1" class="data row11 col1" >1.562500</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row12" class="row_heading level0 row12" >AcceptedCmp2</th>
      <td id="T_c4159_row12_col0" class="data row12 col0" >30</td>
      <td id="T_c4159_row12_col1" class="data row12 col1" >1.339286</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row13" class="row_heading level0 row13" >NumCatalogPurchases</th>
      <td id="T_c4159_row13_col0" class="data row13 col0" >23</td>
      <td id="T_c4159_row13_col1" class="data row13 col1" >1.026786</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row14" class="row_heading level0 row14" >Complain</th>
      <td id="T_c4159_row14_col0" class="data row14 col0" >21</td>
      <td id="T_c4159_row14_col1" class="data row14 col1" >0.937500</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row15" class="row_heading level0 row15" >Income</th>
      <td id="T_c4159_row15_col0" class="data row15 col0" >8</td>
      <td id="T_c4159_row15_col1" class="data row15 col1" >0.357143</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row16" class="row_heading level0 row16" >NumWebVisitsMonth</th>
      <td id="T_c4159_row16_col0" class="data row16 col0" >8</td>
      <td id="T_c4159_row16_col1" class="data row16 col1" >0.357143</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row17" class="row_heading level0 row17" >NumWebPurchases</th>
      <td id="T_c4159_row17_col0" class="data row17 col0" >4</td>
      <td id="T_c4159_row17_col1" class="data row17 col1" >0.178571</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row18" class="row_heading level0 row18" >Age</th>
      <td id="T_c4159_row18_col0" class="data row18 col0" >3</td>
      <td id="T_c4159_row18_col1" class="data row18 col1" >0.133929</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row19" class="row_heading level0 row19" >Total_Spending</th>
      <td id="T_c4159_row19_col0" class="data row19 col0" >3</td>
      <td id="T_c4159_row19_col1" class="data row19 col1" >0.133929</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row20" class="row_heading level0 row20" >Total_Purchases</th>
      <td id="T_c4159_row20_col0" class="data row20 col0" >2</td>
      <td id="T_c4159_row20_col1" class="data row20 col1" >0.089286</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row21" class="row_heading level0 row21" >Customer_Tenure</th>
      <td id="T_c4159_row21_col0" class="data row21 col0" >0</td>
      <td id="T_c4159_row21_col1" class="data row21 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row22" class="row_heading level0 row22" >NumStorePurchases</th>
      <td id="T_c4159_row22_col0" class="data row22 col0" >0</td>
      <td id="T_c4159_row22_col1" class="data row22 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row23" class="row_heading level0 row23" >Kidhome</th>
      <td id="T_c4159_row23_col0" class="data row23 col0" >0</td>
      <td id="T_c4159_row23_col1" class="data row23 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row24" class="row_heading level0 row24" >Recency</th>
      <td id="T_c4159_row24_col0" class="data row24 col0" >0</td>
      <td id="T_c4159_row24_col1" class="data row24 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row25" class="row_heading level0 row25" >Teenhome</th>
      <td id="T_c4159_row25_col0" class="data row25 col0" >0</td>
      <td id="T_c4159_row25_col1" class="data row25 col1" >0.000000</td>
    </tr>
    <tr>
      <th id="T_c4159_level0_row26" class="row_heading level0 row26" >Is_Parent</th>
      <td id="T_c4159_row26_col0" class="data row26 col0" >0</td>
      <td id="T_c4159_row26_col1" class="data row26 col1" >0.000000</td>
    </tr>
  </tbody>
</table>




The table above presents the number of data points identified as outliers via the Interquartile Range (IQR) method for each variable. Additionally, the percentage of outliers in relation to the overall observations is displayed in the second column.

A significant percentage of data labeled as outliers is noticed in several variables. Such outliers may not indeed be outliers but could be representative of the inherent variability within the population parameter for each respective variable. Therefore, caution is advised before deciding to remove these outliers.

Therefore, instead of a blanket removal, a more selective approach will be adopted. Outliers will be defined as those data points where the distance from the nearest data point is significantly large. An example of this can be seen in the 'MntMeatProducts' variable, where values ranging from 1582 to 1725 are significantly distant from the next lower value of 984. This approach will be uniformly applied to all variables.
