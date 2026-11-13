"""Train the logistic regression model and save it for the Streamlit app."""
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("Titanic_train.csv")

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[FEATURES]
y = df["Survived"]

numeric = ["Age", "SibSp", "Parch", "Fare"]
categorical = ["Pclass", "Sex", "Embarked"]

preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

model = Pipeline([("prep", preprocess),
                  ("clf", LogisticRegression(max_iter=1000))])

# Check performance on data the model has not seen
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Hold-out accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, target_names=["Died", "Survived"]))

# Retrain on all the data and save
model.fit(X, y)
joblib.dump(model, "titanic_model.pkl")
print("Saved titanic_model.pkl")
