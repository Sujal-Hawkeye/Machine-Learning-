import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df  = pd.read_csv(r'D:\OneDrive\Documents\homeprices.csv')
print(df.head())
x = df.drop('price', axis=1)
print(x)
y = df.drop('area',axis=1)
print(y)
reg = linear_model.LinearRegression()
print(reg)
print(reg.fit(x, y))
y_pred = reg.predict(x)
print(y_pred)
reg1 = reg.score(x, y)
print('R-squared:', reg1)
print('Coeffecient:', reg.coef_)
print('Intercept:', reg.intercept_)
plt.scatter(x, y, color="green", marker="+")
plt.plot(x, y_pred, color="red", label="Regression line")
plt.legend()
plt.show()
