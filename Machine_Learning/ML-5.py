import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
df = pd.read_csv(r'D:\OneDrive\Documents\diabetes.csv')
print(df.head())
x = df.drop('Outcome', axis='columns')
print(x)
cor = x.corr()
mask = np.triu(np.ones(cor.shape), k=1)
sns.heatmap(x.corr(), mask=mask, annot=True, vmin=-1, vmax=1)
plt.show()
y = df['Outcome']
names_features = x.columns
target_label = y.unique()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(f"Train Data:{len(x_train)}\nTest Data:{len(x_test)}")
model = DecisionTreeClassifier(max_depth=4, random_state=92)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
Accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy:{Accuracy*100:.2f}%")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"classification report:\n{classification_report(y_test, y_pred)}")
confusion_matrix = confusion_matrix(y_test, y_pred)
print(f"confusion matrix:\n{confusion_matrix}")
logmodel = LogisticRegression(max_iter=768)
logmodel.fit(x_train, y_train)
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(x_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area=%0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Recieve operating characterstic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
plt.figure(figsize=(25, 8), facecolor='w', edgecolor='b')
tree.plot_tree(model, feature_names=names_features, rounded=True, filled=True, fontsize=10)
plt.show()
