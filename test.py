import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

df = pd.read_csv("clean_churn.csv")

y = df["Churn"].map({"No": 0, "Yes": 1})

x = df.drop(columns=["Churn", "customerID"])

x = pd.get_dummies(x, dtype=int)

xTrain, xTest, yTrain, yTest = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

decision = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

decision.fit(xTrain, yTrain)

yTrainPredict = decision.predict(xTrain)
yTestPredict = decision.predict(xTest)

# print("Tree depth:", decision.get_depth())
acc = accuracy_score(yTest,yTestPredict)
pr = precision_score(yTest,yTestPredict)
rc = recall_score(yTest,yTestPredict)
fOne = f1_score(yTest,yTestPredict)

print("Acuuracy: ",acc)
print("Precision: ",pr)
print("recall: ",rc)
print("F1 score: ",fOne)


