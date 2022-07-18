import numpy as np
import matplotlib.pyplot as mlt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

iris_df = pd.DataFrame(X_train, columns=iris.feature_names)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

model.predict([[6,3,4,1.5]])

score = model.score(X_test, y_test)

p = score

print(p)