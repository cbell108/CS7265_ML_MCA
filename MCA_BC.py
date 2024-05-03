import sys

from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np

import random

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# data_id = 451         # coimbra
data_id = 17  # wisconcin

use_fetch_ucirepo = True

if use_fetch_ucirepo:
    breast_cancer_dataset = fetch_ucirepo(id=data_id)

    X = pd.DataFrame.from_dict(breast_cancer_dataset.data.features)
    y = pd.DataFrame.from_dict(breast_cancer_dataset.data.targets)

    # Clean wisconcin dataset
    y.replace(to_replace={'M': 1, 'B': 0}, inplace=True)
# variable information
print(X)
print(y)

seed = random.randint(0, 4294967295)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2, random_state=seed)

# Training
lr = LinearRegression()
# lr.fit(X_train, y_train)

dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)

rf = RandomForestClassifier()
# rf.fit(X_train, y_train)

gb = GradientBoostingClassifier()
# gb.fit(X_train, y_train)

models = [lr, dt, rf, gb]
accuracy = []

for model in models:
    model.fit(X_train, y_train)
    y_pred = (model.predict(X_test).squeeze() > .5).astype(int)
    accuracy.append(accuracy_score(y_test, y_pred))

print(f"Accuracy for models lr, dt, rf, gb: {accuracy}")

# Correlation Analysis
corr_matrix = X.corr()

plt.matshow(corr_matrix, cmap=plt.get_cmap('bwr'))
plt.show()

# Find pairs with high correlation

# Drop redundant points
upper = np.triu(np.ones(corr_matrix.shape))
np.fill_diagonal(upper, 0)

threshold = .95
features_corr_matrix = corr_matrix.where(upper.astype(bool)).abs()
corr_features = features_corr_matrix[features_corr_matrix > threshold]

all_corr_features = []
for feature in corr_features.columns:
    list_corr_features = [(feature, corr_features.columns[idx], f) for idx, f in enumerate(corr_features[feature]) if not pd.isna(f)]
    if list_corr_features:
        all_corr_features.extend(list_corr_features)

all_corr_features.sort(key=lambda x: x[2], reverse=True)

# Rank features based on frequency
scores = {}
for pair in all_corr_features:
    feature0 = pair[0]
    feature1 = pair[1]

    if feature0 in scores:
        scores[feature0] += 1
    else:
        scores[feature0] = 0

    if feature1 in scores:
        scores[feature1] += 1
    else:
        scores[feature1] = 0

    # visualization
    print(pair)

print(scores)

drop_columns = [feature for feature, value in scores.items() if value > 0]
trim_X = X.drop(drop_columns, axis=1)

X_train, X_test, y_train, y_test = train_test_split(trim_X, y, train_size=.8, test_size=.2, random_state=seed)

new_accuracy = []
for model in models:
    model.fit(X_train, y_train)
    y_pred = (model.predict(X_test).squeeze() > .5).astype(int)
    new_accuracy.append(accuracy_score(y_test, y_pred))

print(f"New accuracy for models lr, dt, rf, gb: {new_accuracy}")

reductions = [new_accuracy[model]-accuracy[model] for model in range(len(models))]
print(f"Able to remove {len(drop_columns)} columns for change of accuracy: {reductions}")

