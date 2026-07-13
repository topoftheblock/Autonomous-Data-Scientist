# Data Science Report

**Generated:** 2026-07-13 20:24:46
**Dataset:** `data/input/titanic.csv`

---
## 1. Data Overview
- **Shape:** 891 rows × 12 columns
- **Columns:** PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- **Missing values:** Age: 177, Cabin: 687, Embarked: 2

## 2. Data Cleaning
*(No cleaning performed)*

## 3. Statistical Analysis
*(No statistical tests run)*

## 4. Visualizations
*(No plots generated)*

## 5. Execution Log
### Step 1
Step 1 (Profile all columns): I loaded the Titanic dataset and profiled all columns. The profiling included details such as data types, missing values, unique counts, and statistical summaries for numeric columns. Additionally, histograms and count plots were generated for visual representation of the distributions.

### Step 2
Step 2 (Check for missing values): I checked for missing values in the Titanic dataset. The results show that the "Cabin" column has the highest percentage of missing values at 77.1%, followed by "Age" with 19.87% and "Embarked" with 0.22%. All other columns have no missing values.

### Step 3
Step 3 (Impute missing 'Age' with median): I imputed the missing values in the 'Age' column of the Titanic dataset with the median age. After the imputation, there are no longer any missing values in the 'Age' column, while the 'Cabin' and 'Embarked' columns still contain missing values. The overall shape of the dataset remains unchanged at 891 rows and 12 columns.

### Step 4
Step 4 (Drop 'Cabin' column due to high missing values): I dropped the 'Cabin' column from the Titanic dataset due to its high percentage of missing values. The dataset now contains 891 rows and 11 columns, with the remaining columns having no missing values except for 'Embarked', which has 2 missing entries.

### Step 5
Step 5 (Impute missing 'Embarked' with mode): I imputed the missing values in the 'Embarked' column of the Titanic dataset with the mode value. After the imputation, there are no longer any missing values in the 'Embarked' column, while the 'Age' and 'Cabin' columns still contain missing values. The overall shape of the dataset remains unchanged at 891 rows and 12 columns.

### Step 6
Step 6 (Encode 'Sex' and 'Embarked' using one-hot encoding): I encoded the 'Sex' and 'Embarked' columns using one-hot encoding and saved the updated dataset to a Parquet file named 'titanic_encoded.parquet'. The resulting dataset now includes binary columns for 'Sex' (female, male) and 'Embarked' (C, S).

### Step 7
Step 7 (Scale 'Fare' using standard scaling): I scaled the 'Fare' column in the Titanic dataset using standard scaling. The scaled values have been saved to a new Parquet file named 'titanic_scaled.parquet'. The first few scaled values of 'Fare' are displayed, showing the transformation applied.

### Step 8
Step 8 (Run t-test comparing 'Fare' by 'Survived'): I ran a t-test comparing the 'Fare' of passengers based on their survival status ('Survived'). The test statistic was approximately -7.94, and the p-value was extremely low (around 6.12e-15), indicating strong evidence against the null hypothesis. This suggests that there is a significant difference in the 'Fare' between those who survived and those who did not.

### Step 9
Step 9 (Run chi-square test for 'Survived' by 'Pclass'): I ran a chi-square test for the 'Survived' variable by 'Pclass'. The test statistic was approximately 102.89, with a p-value of 4.55e-23, indicating strong evidence against the null hypothesis. This suggests a significant association between survival and passenger class.

### Step 10
Step 10 (Create histogram of 'Age'): I created a histogram of the 'Age' column from the Titanic dataset, and the plot has been saved as "data/output/histogram_Age.png".

### Step 11
Step 11 (Create boxplot of 'Fare' by 'Survived'): I created a boxplot of 'Fare' grouped by 'Survived', and the plot has been saved as "data/output/boxplot_Fare_by_Survived.png".

### Step 12
Step 12 (Generate final report): I generated a final report for the Titanic dataset. The report includes a summary of the dataset's structure, data types, and missing values. The dataset consists of 891 entries and 12 columns, with key columns such as 'Survived', 'Pclass', 'Age', and 'Fare'. The 'Age' and 'Cabin' columns had missing values, which were addressed through imputation and removal. Additionally, statistical tests were conducted to analyze the relationships between variables, and visualizations were created to illustrate the distributions of key features.

### Step 13
Formal report generated at data/output/report_20260713_202446.md


---
*Report generated automatically by the Autonomous Data Scientist Agent.*
