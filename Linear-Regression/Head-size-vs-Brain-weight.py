# Importing Necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


# Reading Data
df = pd.read_csv('/content/head_brain.csv')
df.head()

df.info()

df.isnull().sum()
# Our dataset has no categorical values we can move forward.
# we don't have any null values in our dataset.

df.shape

# Collecting x and y variables
X = df['Head Size(cm^3)'].values
Y =  df['Brain Weight(grams)'].values

X.shape, Y.shape

# Method 1: Manual Coding
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Total number of values
n = len(X)

numerator = 0
denominator = 0

# Using the formula to calculate m and c
for i in range(n):
    numerator += (X[i]-mean_X)* (Y[i]-mean_Y)
    denominator +=(X[i]-mean_X)**2
m = numerator/denominator
c = mean_Y - (m*mean_X)

# Print coefficients
print(m, c)


plt.scatter(X, Y)

min_x = np.min(X) - 100
max_x = np.max(X) + 100

# Calculating line values x and y
x = np.linspace(min_x, max_x, 1000)

y = m * x + c

#plotting Scatter Points
plt.scatter(X,Y,color='b', label='Scatter Plot')

# Plotting line
plt.plot(x,y,color='r', label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Head size cm^3')
plt.ylabel('Brain weight in grams')

sum_pred = 0
sum_act = 0

for i in range(n):
    y_pred = (m*X[i]+c)
    sum_pred += (Y[i]-y_pred)**2
    sum_act +=(Y[i]-mean_Y)**2

r2 = 1-(sum_pred/sum_act)
print(r2) # Here we can observe that we got R**2> 0.5 . so we have good model

def predict(x):
    y = m*x + c
    print(y)

predict(4146)

# Method 2: using scikit learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X  = X.reshape((n,1))

X.shape
y.shape

lg = LinearRegression()
lg.fit(X,Y)

# Y prediction
y_pred = lg.predict(X)

mse = mean_squared_error(Y,y_pred)
rmse = np.sqrt(mse)

# Calculating R2 Score
r2_score = lg.score(X,Y)

print(rmse)
print(r2_score)
# we got the same error R**2 value as above method-1

lg.predict([[4146]])
lg.intercept_
