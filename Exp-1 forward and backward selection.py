import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score

wine = pd.read_csv("Wine dataset.csv")

X = wine.drop("target", axis=1)
y = wine["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

forward = SequentialFeatureSelector(
    model,
    n_features_to_select=5,
    direction="forward"
)

forward.fit(X_train, y_train)

X_train_forward = forward.transform(X_train)
X_test_forward = forward.transform(X_test)

model.fit(X_train_forward, y_train)
y_pred = model.predict(X_test_forward)

print("Selected Features using Forward Selection")
print(X.columns[forward.get_support()])
print("Forward Selection Accuracy:")
print(accuracy_score(y_test, y_pred))

backward = SequentialFeatureSelector(
    model,
    n_features_to_select=5,
    direction="backward"
)

backward.fit(X_train, y_train)

X_train_backward = backward.transform(X_train)
X_test_backward = backward.transform(X_test)

model.fit(X_train_backward, y_train)
y_pred = model.predict(X_test_backward)

print("\nSelected Features using Backward Selection")
print(X.columns[backward.get_support()])
print("Backward Selection Accuracy:")
print(accuracy_score(y_test, y_pred))
