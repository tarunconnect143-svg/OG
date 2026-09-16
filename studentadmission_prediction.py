import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("admission.csv")

X = data.drop(["Serial No.", "Chance of Admit "], axis=1)
y = data["Chance of Admit "]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Student Admission Prediction")
print("----------------------------")
print("R2 Score:", round(r2_score(y_test, y_pred), 4))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 4))

gre = float(input("Enter GRE Score: "))
toefl = float(input("Enter TOEFL Score: "))
rating = float(input("Enter University Rating: "))
sop = float(input("Enter SOP Strength: "))
lor = float(input("Enter LOR Strength: "))
cgpa = float(input("Enter CGPA: "))
research = float(input("Research Experience (0/1): "))

student = [[gre, toefl, rating, sop, lor, cgpa, research]]
prediction = model.predict(student)[0]

print("Predicted Chance of Admission:", round(prediction, 4))