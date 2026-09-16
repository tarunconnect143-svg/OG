import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Credit Card Fraud Detection")
print("---------------------------")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

amount = float(input("Enter transaction amount: "))

transaction = X.iloc[[0]].copy()
transaction["Amount"] = scaler.transform([[amount]])[0][0]

prediction = model.predict(transaction)[0]

if prediction == 1:
    print("Result: Fraudulent Transaction")
else:
    print("Result: Legitimate Transaction")