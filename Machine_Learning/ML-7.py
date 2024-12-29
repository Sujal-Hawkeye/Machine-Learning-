import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

df = load_digits()

df_num = pd.DataFrame(df.data, columns=df.feature_names)
x_df = df_num
y_df = df.target
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_df)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_df, test_size=0.2, random_state=30)
print("Training Set:", len(x_train))
print("Testing Set", len(x_test))

model = LogisticRegression(max_iter=170)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("Accuracy score without PCA:", score)
pca_df = PCA(0.95)
x_pca = pca_df.fit_transform(x_scaled)
print("Number of components with 95% variance explained:", pca_df.n_components_)
print("Explained variance ratio:", pca_df.explained_variance_ratio_)
x_train_pca, x_test_pca, y_train, y_test = train_test_split(x_pca, y_df, test_size=0.2, random_state=30)
model = LogisticRegression(max_iter=170)
model.fit(x_train_pca, y_train)
score_pca = model.score(x_test_pca, y_test)
print("Accuracy score with PCA:", score_pca)

plt.figure(figsize=(8, 6))
for i in range(10):
    plt.scatter(x_pca[y_df == i, 0], x_pca[y_df == i, 1], label=str(i))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot after PCA (2D)')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(10):
    ax.scatter(x_pca[y_df == i, 0], x_pca[y_df == i, 1], x_pca[y_df == i, 2], label=str(i))
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Scatter Plot after PCA (3D)')
plt.legend()
plt.show()
print("Shape of x_df (Original Data):", x_df.shape)
print("Shape of x_scaled (Standardized Data):", x_scaled.shape)
print("Shape of x_pca (PCA Reduced Data):", x_pca.shape)
