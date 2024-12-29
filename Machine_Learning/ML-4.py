import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # only using the first two features for visualization
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define SVM classifiers with different kernels
C = 1.0  # SVM regularization parameter
classifiers = {
    "Linear SVM": svm.SVC(kernel="linear", C=C),
    "LinearSVC": svm.LinearSVC(C=C, max_iter=10000, dual=False),
    "RBF SVM": svm.SVC(kernel="rbf", gamma=0.7, C=C),
    "Polynomial SVM": svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
}

# Train and make predictions for each classifier
predictions = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    predictions[name] = y_pred

# Calculate accuracy for each classifier
accuracies = {}
for name, y_pred in predictions.items():
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc

# Print the data table (first 15 rows)
data_df = pd.DataFrame(np.column_stack([X_test[:15], y_test[:15]]), columns=iris.feature_names[:2] + ["target"])
print("Test Data Table (First 15 Rows):")
print(data_df)

# Print the accuracy for each classifier
print("\nAccuracy of SVM Classifiers:")
for name, acc in accuracies.items():
    print(f"{name}: {acc:.2f}")

# Plot decision boundaries for each classifier
plt.figure(figsize=(12, 10))
plt.title('Decision Boundaries for SVM Classifiers')
colors = ['red', 'green', 'blue']
for idx, (name, clf) in enumerate(classifiers.items()):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, colors=colors)
    plt.scatter(X_test[:15, 0], X_test[:15, 1], c=y_test[:15], cmap=plt.cm.Paired, s=20, edgecolors="k", label="Test Data")
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.xticks(())
    plt.yticks(())
    plt.title(name)
    plt.legend()
plt.tight_layout()
plt.show()
