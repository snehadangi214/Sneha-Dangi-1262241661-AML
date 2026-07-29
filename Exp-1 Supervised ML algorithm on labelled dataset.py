import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

wine = pd.read_csv("Wine dataset.csv")

X = wine.drop("target", axis=1)
y = wine["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Decision Tree Accuracy:")
print(accuracy_score(y_test, y_pred))
