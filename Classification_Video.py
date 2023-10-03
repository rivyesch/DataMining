import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import seaborn as sns
import numpy as np

# importing the data
df_features = pd.read_csv("WLA.csv", header=None, names=["Width (W)", "Length (L)", "Area (A)"])
df_labels = pd.read_csv("Labels.csv", header=None, names=["True Label"])

# combining the features and the labels in a single dataframe
df = pd.concat([df_features, df_labels], axis=1)
# print(df.shape)

# Removing the first 16 lines since it contains only one of the two objects of interest - motorbike
df = df.iloc[16:]
# print(df.shape)

# Normalise - converts all values in the columns from 0 to 1
mn = MinMaxScaler()
norm = mn.fit_transform(df)
df_norm = pd.DataFrame(norm, columns=df.columns)
# print(df_norm.head())

# Visualising the data distribution for the car and motorcycle
plt.figure()
sns.kdeplot(df_norm.loc[df_norm["True Label"] == 0, "Length (L)"], label=0)
sns.kdeplot(df_norm.loc[df_norm["True Label"] == 1, "Length (L)"], label=1)
plt.savefig('Length Distribution.png', dpi=600)

plt.figure()
sns.kdeplot(df_norm.loc[df_norm["True Label"] == 0, "Width (W)"], label=0)
sns.kdeplot(df_norm.loc[df_norm["True Label"] == 1, "Width (W)"], label=1)
plt.savefig('Width Distribution.png', dpi=600)

plt.figure()
sns.kdeplot(df_norm.loc[df_norm["True Label"] == 0, "Area (A)"], label=0)
sns.kdeplot(df_norm.loc[df_norm["True Label"] == 1, "Area (A)"], label=1)
plt.savefig('Area Distribution.png', dpi=600)
plt.show()

# Plotting histogram to observe the distribution of the data for each variable (feature and target)
# fig = plt.figure(figsize=(10, 7))
# i = 0
# for column in df_norm:
#     sub = fig.add_subplot(2, 2, i+1)
#     sub.set_xlabel(column)
#     df_norm[column].plot(kind="hist")
#     i += 1
#
# plt.show()

# Splitting the dataframe to independent variables(X) and dependent variables(y)
X = df_norm.drop('True Label', axis=1)
y = df_norm['True Label']

# Split dataset into training and testing set(33%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=66, shuffle=True)

########################################################################################################################

# Machine learning model - random forests
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, rfc_pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(y_test, rfc_pred))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())

# Only needs to be done once to find the optimal hyperparameters
# # number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# # number of features at every split
# max_features = ['auto', 'sqrt']
#
# # max depth
# max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
# max_depth.append(None)
# # create random grid
# random_grid = {
#     'n_estimators': n_estimators,
#     'max_features': max_features,
#     'max_depth': max_depth
#  }
#
# # Random search of parameters
# rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
#                                 random_state=42, n_jobs=-1)
#
# # Fit the model
# rfc_random.fit(X_train, y_train)
# # print results
# print(rfc_random.best_params_)

# Optimise the model
rfc = RandomForestClassifier(n_estimators=1000, max_depth=140, max_features='auto')
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')

print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))

########################################################################################################################

# Machine learning model - SVM
clf = svm.SVC(kernel="linear")
clf_trained = clf.fit(X_train, y_train)
print(clf_trained.score(X_train, y_train))

# Finding the best hyperparameters
params = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

clf = GridSearchCV(
    estimator=svm.SVC(),
    param_grid=params,
    cv=10,
    n_jobs=5,
    verbose=1
)

clf.fit(X_train, y_train)
print(clf.best_params_)

# Building and fit the classifier
clf = svm.SVC(kernel='rbf', gamma=1, C=0.1)
clf_trained = clf.fit(X_train, y_train)

# Get support vector indices
support_vector_indices = clf.support_
print(support_vector_indices)

# Get number of support vectors per class
support_vectors_per_class = clf.n_support_
print(support_vectors_per_class)

# Get support vectors themselves
support_vectors = clf.support_vectors_

# Make predictions and check the accuracy
y_pred = clf_trained.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

