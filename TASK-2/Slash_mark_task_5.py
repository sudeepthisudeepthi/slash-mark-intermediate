import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("BlackFridaySales.csv")

# Convert the "Purchase" column to numeric
data['Purchase'] = pd.to_numeric(data['Purchase'], errors='coerce')

# Drop rows with missing Purchase values
data.dropna(subset=['Purchase'], inplace=True)

# Continue with your analysis...

sns.distplot(data["Purchase"], color='r')
plt.title("Purchase Distribution")
plt.show()

sns.boxplot(data["Purchase"])
plt.title("Boxplot of Purchase")
plt.show()

print("Skewness:", data["Purchase"].skew())
print("Kurtosis:", data["Purchase"].kurtosis())
print("Description:\n", data["Purchase"].describe())

sns.countplot(data['Gender'])
plt.show()

print("Percentage of Gender Distribution:")
print(data['Gender'].value_counts(normalize=True) * 100)

# Convert "Purchase" column back to numeric for calculations
data['Purchase'] = pd.to_numeric(data['Purchase'])

# Mean Purchase by Gender
mean_purchase_by_gender = data.groupby("Gender")["Purchase"].mean()
print("Mean Purchase by Gender:")
print(mean_purchase_by_gender)

# Continue with the rest of your analysis...
print(data['Marital_Status'].dtype)
print(data['Marital_Status'].unique())
mean_purchase_by_gender = data.groupby("Gender")["Purchase"].mean()
print("Mean Purchase by Gender:")
print(mean_purchase_by_gender)
# Check the unique values of 'Marital_Status' again
print(data['Marital_Status'].unique())
# Plot count plot for 'Marital_Status'
sns.countplot(data['Marital_Status'])
plt.title("Marital Status Distribution")
plt.show()