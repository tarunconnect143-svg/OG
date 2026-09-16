import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("concrete_data.csv")

X = data.drop("Strength", axis=1)
y = data["Strength"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Concrete Compressive Strength Prediction")
print("-----------------------------------------")
print("R2 Score:", round(r2_score(y_test, y_pred), 4))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 4))

cement = float(input("Enter Cement quantity: "))
water = float(input("Enter Water quantity: "))
age = float(input("Enter Age of concrete: "))

sample = X.iloc[[0]].copy()
sample.iloc[0, :] = X.iloc[0, :].values

sample["Cement"] = cement
sample["Water"] = water
sample["Age"] = age

prediction = model.predict(sample)[0]

print("Predicted Compressive Strength:", round(prediction, 2))