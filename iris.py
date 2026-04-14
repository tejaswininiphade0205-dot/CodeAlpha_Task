# Import libraries
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

dataf = pd.read_csv("Iris.csv")


dataf.drop("Id", axis=1, inplace=True)


dataf.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("First 5 Rows:\n", dataf.head())


dataf['species'] = dataf['species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

print("\nTarget Mapping:")
print("0 = Setosa, 1 = Versicolor, 2 = Virginica")


X = dataf[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = dataf['species']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#use new input
sample = pd.DataFrame([[7.1, 4.5, 2.4, 6.2]], columns=X.columns)

prediction = model.predict(sample)

species_names = ['Setosa', 'Versicolor', 'Virginica']

print("\nPredicted Class:", prediction[0])
print("Predicted Species:", species_names[prediction[0]])