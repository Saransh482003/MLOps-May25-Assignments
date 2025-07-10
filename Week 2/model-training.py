import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('data/iris.csv')
data.head(5)

data.describe()

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

dec_clf = DecisionTreeClassifier(max_depth = 3, random_state = 1)
dec_clf.fit(X_train,y_train)
y_pred = dec_clf.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(accuracy_score(y_test,y_pred)))


print(classification_report(y_test, y_pred))
