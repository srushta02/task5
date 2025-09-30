```python
# Titanic EDA - Exploratory Data Analysis
# Tools: Pandas, Matplotlib, Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# ------------------------------
# Basic Exploration
# ------------------------------
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include="all"))

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Value counts for categorical features
print("\nSurvived Distribution:")
print(df['Survived'].value_counts())

print("\nPclass Distribution:")
print(df['Pclass'].value_counts())

print("\nSex Distribution:")
print(df['Sex'].value_counts())

# ------------------------------
# Visualizations
# ------------------------------
sns.set(style="whitegrid")

# 1. Histogram of numerical features
df.hist(figsize=(12, 8), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.show()

# 2. Countplot of survival
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df, palette="Set2")
plt.title("Survival Count")
plt.show()

# 3. Countplot of survival by gender
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", hue="Sex", data=df, palette="Set1")
plt.title("Survival by Gender")
plt.show()

# 4. Survival by class
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", hue="Pclass", data=df, palette="Set3")
plt.title("Survival by Passenger Class")
plt.show()

# 5. Boxplot of Age vs Survival
plt.figure(figsize=(6,4))
sns.boxplot(x="Survived", y="Age", data=df, palette="Set2")
plt.title("Age Distribution by Survival")
plt.show()

# 6. Scatterplot of Age vs Fare, colored by Survival
plt.figure(figsize=(8,6))
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=df, palette="coolwarm")
plt.title("Age vs Fare (Survival)")
plt.show()

# 7. Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 8. Pairplot (subset of features)
sns.pairplot(df[["Survived", "Age", "Fare", "Pclass"]], hue="Survived", palette="husl")
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()

# ------------------------------
# Observations (example to include in PDF/Report)
# ------------------------------
"""
- Majority of passengers did not survive (about 62%).
- Females had a much higher survival rate than males.
- Higher-class passengers (Pclass=1) had better survival chances.
- Younger passengers and children had slightly better survival rates.
- Fare shows a positive correlation with survival (wealthier passengers survived more).
"""
```
