import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# import csv and categorize x & y

df = pd.read_csv('btc.csv')
x = df['number']
Y = df['close']

# ask what to predict

inital_str = input("what day do you want to predict: ")
inital = float(inital_str)

# creates 3 models

polynomial_regression = np.poly1d(np.polyfit(x,Y,7))
polynomial_regression_min = np.poly1d(np.polyfit(x,Y,6))
polynomial_regression_max = np.poly1d(np.polyfit(x,Y,9))

# plot main csv graphed and make beautiful line

plt.plot(x,Y)
xp = np.linspace(0,93,100)

# print predicted values

print("maximum value: "+str(polynomial_regression_max(inital)))
print("minimum value: "+str(polynomial_regression_min(inital)))
print("middle value: "+str(polynomial_regression(inital)))

# find mean of predicted values (4th and final prediction)

predicted_values_raw = [polynomial_regression(inital),polynomial_regression_max(inital),polynomial_regression_min(inital)]
mean_val = np.mean(predicted_values_raw)
print(f"mean of all three: {str(mean_val)}")

# plot the polynomial regression lines onto the graph

plt.plot(xp, polynomial_regression(xp), c='r')
plt.plot(xp, polynomial_regression_min(xp), c='g')
plt.plot(xp, polynomial_regression_max(xp), c='y')

# plot the predicted values onto the graph

plt.scatter(inital, polynomial_regression_min(inital),c='g')
plt.scatter(inital, polynomial_regression_max(inital),c='y')
plt.scatter(inital, polynomial_regression(inital), c='r')
plt.scatter(inital, np.mean(predicted_values_raw))

# show graph

plt.show()

#
# 1 WEEK TESTING PREDICTIONS FOR FINDING BEST DEGREE OR IF MEAN SOLOS
# 1 WEEK TESTING PREDICTIONS FOR FINDING BEST DEGREE OR IF MEAN SOLOS
# 1 WEEK TESTING PREDICTIONS FOR FINDING BEST DEGREE OR IF MEAN SOLOS
#

# 15 Oct prediction:
#
#actual closing value: 19067
#close from day before: 19185
#
#closest: 19036 (mean)
#
#maximum value: 18109.04807879844
#minimum value: 19340.0416496139
#middle value: 19659.236966181903
#mean of all three: 19036.108898198083
#
# actually closer than expected, small down predicted
#

