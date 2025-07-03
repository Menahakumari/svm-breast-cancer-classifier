import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/breast-cancer.csv')
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

if y.dtype == 'object':
    y = y.astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train SVM with Linear and RBF Kernels
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# Linear Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
print("SVM with Linear Kernel:\n", classification_report(y_test, y_pred_linear))
# RBF Kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
print("SVM with RBF Kernel:\n", classification_report(y_test, y_pred_rbf))
# Visualize Decision Boundary (only for 2D data)
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y, title):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Use only 2 features for visualization
X_2d = X.iloc[:, :2]  # You can choose other pairs too
X_2d_scaled = StandardScaler().fit_transform(X_2d)
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d_scaled, y, test_size=0.2, random_state=42)

model_2d = SVC(kernel='rbf', C=1)
model_2d.fit(X_train_2d, y_train_2d)

plot_decision_boundary(model_2d, X_train_2d, y_train_2d, "SVM Decision Boundary (2D, RBF Kernel)")
#Tune Hyperparameters (C, gamma)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
# Cross-Validation Performance
from sklearn.model_selection import cross_val_score

best_svm = grid.best_estimator_
cv_scores = cross_val_score(best_svm, X, y, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
