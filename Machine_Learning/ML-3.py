import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
import seaborn as sb
df = pd.read_csv(r'D:\OneDrive\Documents\diabetes.csv')
print(df)
x = df.drop('Outcome', axis='columns')
print(x)
cor = x.corr()
mask = np.triu(np.ones_like(cor))
sb.heatmap(x.corr(), annot=True, vmin=-1, vmax=1, mask=mask)
plt.show()
y = df.Outcome
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print(f"Train Data:{len(X_train)}\nTest Data:{len(X_test)}")
model = LogisticRegression(max_iter=786)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy:{accuracy*100:.2f}%')
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification_report:\n", classification_report(y_test, y_pred))
cf_mtrx = confusion_matrix(y_test, y_pred)
x_labels = ['False', 'True']
y_labels = ['False', 'True']
group_names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
group_counts = ["{:.0f}".format(value) for value in cf_mtrx.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
labels = np.asarray(labels).reshape(2,2)
sb.heatmap(cf_mtrx, annot=labels, fmt='', cmap='flare', xticklabels=x_labels, yticklabels=y_labels)
plt.xlabel('Predicted Diagnosis')
plt.ylabel('Actual Diagnosis')
plt.title('Confusion Matrix for\n Heart Disease Detection Model')
plt.show()
logmodel = LogisticRegression(max_iter=768)
logmodel.fit(X_train, y_train)
logit_roc_auc = roc_auc_score(y_test, logmodel.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logmodel.predict_proba(X_test)[:, 1])
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

