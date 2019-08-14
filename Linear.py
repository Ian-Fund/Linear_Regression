import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Ecommerce Customers")

print(customers.head())

print(customers.describe())

# Data Exploration

sns.jointplot(x='Time on Website',y='Yearly Amount Spent', data=customers)
plt.show()


sns.jointplot(x='Time on App',y='Yearly Amount Spent', data=customers)
plt.show()

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex', data=customers)
plt.show()

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
plt.show()

# Linear Regression

y = customers['Yearly Amount Spent']
x = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train,y_train)

print('Coefficients: \n', lm.coef_)

predictions = lm.predict(x_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test,predictions))
print('MSE: ', metrics.mean_squared_error(y_test,predictions))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test,predictions)))

coefficients = pd.DataFrame(lm.coef_,x.columns)
coefficients.columns = ['Coeffcient']
print(coefficients)