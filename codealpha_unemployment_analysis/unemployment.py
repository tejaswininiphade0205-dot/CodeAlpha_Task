import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataf = pd.read_csv("unemployment.csv")

# Data Cleaning
dataf.dropna(inplace=True)
dataf.columns = dataf.columns.str.strip()
dataf['Date'] = pd.to_datetime(dataf['Date'], dayfirst=True)


print(dataf.head())
print(dataf.info())


print("\nStatistical Summary:\n", dataf.describe())

print("\nRegions:\n", dataf['Region'].unique())

print("\nArea Distribution:\n", dataf['Area'].value_counts())


plt.figure(figsize=(10,5))
plt.plot(dataf['Date'], dataf['Estimated Unemployment Rate (%)'], marker='o')
plt.xticks(rotation=45)
plt.title("Unemployment Trend")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.grid()
plt.show()


#COVID-19 Analysis

covid_data = dataf[(dataf['Date'].dt.year == 2020) | (dataf['Date'].dt.year == 2021)]

plt.figure(figsize=(10,5))
plt.plot(covid_data['Date'], covid_data['Estimated Unemployment Rate (%)'], color='red', marker='o')
plt.title("Covid Impact on Unemployment")
plt.xlabel("Date")
plt.ylabel("Rate (%)")
plt.xticks(rotation=45)
plt.grid()
plt.show()




region_avg = dataf.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,6))
region_avg.head(10).plot(kind='bar')
plt.title("Top 10 Regions with Highest Unemployment")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.show()


area_avg = dataf.groupby('Area')['Estimated Unemployment Rate (%)'].mean()

plt.figure()
area_avg.plot(kind='bar')
plt.title("Rural vs Urban Unemployment")
plt.xlabel("Area")
plt.ylabel("Unemployment Rate (%)")
plt.show()


dataf['Month'] = dataf['Date'].dt.month

plt.figure(figsize=(10,5))
sns.boxplot(x='Month', y='Estimated Unemployment Rate (%)', data=dataf)
plt.title("Monthly Unemployment Pattern")
plt.xlabel("Month")
plt.ylabel("Rate (%)")
plt.show()
