import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "area": [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    "age": [10, 8, 7, 5, 5, 4, 3, 2, 2, 1],
    "price": [50, 58, 70, 82, 90, 105, 120, 135, 150, 180]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms", "age"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)

parameters = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 4]
}

grid = GridSearchCV(
    model,
    parameters,
    cv=3,
    scoring="r2"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

print("Best Parameters:", grid.best_params_)
print("Predicted Prices:", predictions)
print("Actual Prices:", list(y_test))
print("R2 Score:", r2_score(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)