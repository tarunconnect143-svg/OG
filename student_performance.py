import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = {
    "study_hours": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "attendance": [60, 65, 70, 75, 80, 82, 85, 90, 92, 95],
    "previous_score": [50, 55, 60, 65, 70, 72, 75, 80, 85, 90],
    "final_score": [45, 52, 58, 65, 70, 75, 78, 85, 88, 94]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)

parameters = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, None]
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
print("Predicted Scores:", predictions)
print("Actual Scores:", list(y_test))
print("R2 Score:", r2_score(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)