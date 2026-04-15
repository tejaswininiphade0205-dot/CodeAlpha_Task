
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("car_data.csv")



print(df.head())
print(df.info())

 
df.dropna(inplace=True)

 
df = pd.get_dummies(df, drop_first=True)


df['Car_Age'] = 2026 - df['Year']


df.drop('Year', axis=1, inplace=True)


X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

 
print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction")
plt.show()

 
sample = X_test.iloc[[0]]
print("\nPredicted Price:", model.predict(sample))