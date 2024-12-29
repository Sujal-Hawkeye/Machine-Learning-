import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sns
x = pd.read_csv(r'D:\OneDrive\Documents\income.csv')
print(x)
x_train, x_test = train_test_split(x[['Age', 'Income($)']], test_size=0.33, random_state=0)
scaler = preprocessing.StandardScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)
K = range(2, 10)
inertia = []
for k in K:
    model = KMeans(n_clusters=k, random_state=0)
    model.fit(x_train_norm)
    inertia.append(model.inertia_)
plt.figure()
sns.lineplot(x=K, y=inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=0)
kmeans.fit(x_train_norm)
x_train['Cluster'] = kmeans.labels_

sns.scatterplot(data=x_train, x='Age', y='Income($)', hue='Cluster', palette='Set1')
plt.title('Income vs Age with Clusters')
plt.show()

