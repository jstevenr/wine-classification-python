# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import from scikit-learn
import sklearn.metrics as metrics
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from statistics import mean
# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')

# reading in the data
df = pd.read_csv("winequality-white.csv", sep = ";")
print(df.isnull().head()) # check for missing values

# correlation matrix
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# dropping residual.sugar, density, and total.sulfur.dioxide
df.drop(["residual sugar", "density", "total sulfur dioxide"], axis = 1,
        inplace = True)

# new correlation matrix
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# assigning the independent  variables to X
X = df.loc[:, 'fixed acidity':'alcohol']


# converting the dependent variable from numeric to categorical
def score_to_label(x):
    if x > 5:
        return "good"
    else:
        return "bad"

# replacing the numeric 'quality' with categorical 'label'
df['label'] = df['quality'].apply(score_to_label)
df.drop(['quality'], axis = 1, inplace = True)
df.label, class_names = pd.factorize(df.label)
y = df.label
# split the data intro training and test sets, test size is 30% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

# scaling the numeric attributes
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


### Model Fitting ####
# decision tree classifier object
tree_clf = DecisionTreeClassifier(criterion = "gini", random_state = 100)
tree_clf.fit(X_train, y_train) # training the model
tree_y_pred = tree_clf.predict(X_test) # predicting

def calculate_roc(clf, X_test, y_test):
    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def plot_roc_curve(clf, X_test, y_test, name):
    probs = clf.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    title = 'ROC Curve, ' + name
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# metrics
tree_acc = metrics.accuracy_score(y_test, tree_y_pred)
tree_misclass = mean(tree_y_pred != y_test) # misclassification error
tree_roc = calculate_roc(tree_clf, X_test, y_test)

plot_roc_curve(tree_clf, X_test, y_test, "Decision Tree")

# 10-fold cross validation to find the optimal size of the tree
kf = KFold(n_splits = 10, shuffle = True, random_state = 100)
cv_results = cross_val_score(estimator = tree_clf, X = X, y = y, cv = kf,
                             scoring = "accuracy", n_jobs = -1)

print("mean: {:.3f} (std: {:.3f})".format(cv_results.mean(),
                                          cv_results.std()),
                                          end="\n\n" )
# k-Fold cross validation
# k = 10
depth = []
kf = KFold(n_splits = 10, shuffle = True, random_state = 100)
for i in range(3,20):
    test_clf = DecisionTreeClassifier(criterion = 'gini', max_depth=i)
    test_clf.fit(X_train, y_train) # training the model
    test_y_pred = test_clf.predict(X_test) # making predictions
    misclass = mean(test_y_pred != y_test) # misclassification error
    scores = cross_val_score(estimator = test_clf, X=X, y=y, cv=10, n_jobs=-1)
    depth.append((i, misclass, scores.mean()))

# Comparing number of leaves to misclassification error
depth = pd.DataFrame(depth)
depth.columns = ['leaves','misclass_error', 'CV_score']

plt.scatter(depth.leaves, depth.misclass_error)
plt.title('Number of leaves vs misclassification error')
plt.xlabel('Number of leaves')
plt.ylabel('Misclassification error')

# Cross validation score
plt.scatter(depth.leaves, depth.CV_score)
plt.title('Number of leaves vs Cross Validation score')
plt.xlabel('Number of leaves')
plt.ylabel('Cross validation score')

# the optimal number of leaves is 6
pruned_tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 6)
pruned_tree.fit(X_train, y_train)
pruned_y_pred = pruned_tree.predict(X_test)

# metrics
pruned_acc = metrics.accuracy_score(y_test, pruned_y_pred)
print(pruned_acc)
print(metrics.classification_report(y_test, pruned_y_pred))
pruned_misclass = mean(pruned_y_pred != y_test) # misclassification error
print(pruned_misclass)
pd.DataFrame(metrics.confusion_matrix(y_test, pruned_y_pred),
             columns = ['Predicted Bad', 'Predicted Good'],
             index = ['True Bad', 'True Good'])

# calculate the fpr and tpr for all thresholds of the classification
pruned_probs = pruned_tree.predict_proba(X_test)
pruned_preds = pruned_probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, pruned_preds)
pruned_roc_auc = metrics.auc(fpr, tpr)
print(pruned_roc_auc)

# plotting with plt
plt.title('Receiver Operating Characteristic (after Pruning)')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % pruned_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# k-nearest neighbors
clf_knn = neighbors.KNeighborsClassifier(10, weights = "distance")
clf_knn.fit(X_train, y_train)
knn_y_pred = clf_knn.predict(X_test)

# metrics
knn_acc = metrics.accuracy_score(y_test, knn_y_pred)
print(knn_acc)
print(metrics.classification_report(y_test, knn_y_pred))
knn_misclass = mean(knn_y_pred != y_test) # misclassification error
print(knn_misclass)
pd.DataFrame(metrics.confusion_matrix(y_test, knn_y_pred),
             columns = ['Predicted Bad', 'Predicted Good'],
             index = ['True Bad', 'True Good'])

# calculate the fpr and tpr for all thresholds of the classification
knn_probs = clf_knn.predict_proba(X_test)
knn_preds = knn_probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, knn_preds)
knn_roc_auc = metrics.auc(fpr, tpr)
print(knn_roc_auc)

# plotting with plt
plt.title('Receiver Operating Characteristic, k-Nearest Neighbors')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % knn_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# initialize dataFrame to store accuracy rates and AUC for different values of k
knn_df = []
for k in range(1,21):
    # fitting and using the knn classifier
    test_clf_knn = neighbors.KNeighborsClassifier(n_neighbors = k, weights = "distance") # creating the classifier
    test_clf_knn.fit(X_train, y_train) # fitting the classifier
    test_knn_pred = test_clf_knn.predict(X_test) # predicting with the test set

    # Calculate AUC
    test_probs = test_clf_knn.predict_proba(X_test)
    test_preds = test_probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, test_preds)
    test_roc_auc = metrics.auc(fpr, tpr)

    knn_df.append({'k': k,
               'AccuracyRate': metrics.accuracy_score(y_test, test_knn_pred),
               'AUC': test_roc_auc,
               'MisclassRate': np.mean(test_knn_pred != y_test)})

knn_df = pd.DataFrame(knn_df)
knn_df.head()

# plotting
plt.scatter(knn_df['k'], knn_df['AccuracyRate'])
plt.title('k-nearest neighbors vs Accuracy Rate')
plt.xlabel('k')
plt.ylabel('Accuracy Rate')
plt.show()

plt.scatter(knn_df['k'], knn_df['AUC'])
plt.title('k-nearest neighbors vs AUC')
plt.xlabel('k')
plt.ylabel('AUC')
plt.show()

plt.scatter(knn_df['k'], knn_df['MisclassRate'])
plt.title('k-nearest neighbors vs Misclassification Rate')
plt.xlabel('k')
plt.ylabel('Misclassification Rate')
plt.show()

# optimal seems to be k = 10, our original choice

# randomForest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(bootstrap = True, criterion = 'gini', n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_y_pred = rf_clf.predict(X_test)

# metrics
rf_acc = metrics.accuracy_score(y_test, rf_y_pred)
print(rf_acc)
print(metrics.classification_report(y_test, rf_y_pred))
rf_misclass = mean(rf_y_pred != y_test) # misclassification error
print(rf_misclass)
pd.DataFrame(metrics.confusion_matrix(y_test, rf_y_pred),
             columns = ['Predicted Bad', 'Predicted Good'],
             index = ['True Bad', 'True Good'])

calculate_roc(rf_clf, X_test, y_test)
plot_roc_curve(rf_clf, X_test, y_test, name = "random Forest")


# calculate the fpr and tpr for all thresholds of the classification
rf_probs = rf_clf.predict_proba(X_test)
rf_preds = rf_probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, rf_preds)
rf_roc_auc = metrics.auc(fpr, tpr)
print(rf_roc_auc)

# plotting with plt
plt.title('Receiver Operating Characteristic, k-Nearest Neighbors')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % rf_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()