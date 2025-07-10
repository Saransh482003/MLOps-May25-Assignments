import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    return pd.read_csv('data/iris.csv')

def split_data(data):
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train['species']
    X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test['species']
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

import joblib
def save_model(model, path="model.pkl"):
    joblib.dump(model, path)
