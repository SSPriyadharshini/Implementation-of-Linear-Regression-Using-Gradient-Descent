# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:
### Profit Prediction Graph :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/d64e3ca6-c94d-49b5-9116-a817b1d6d623)

![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/92625829-e1c6-473f-8f6a-f00d6209bdd6)
### Compute Cost Value :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/9e384697-fc9f-4277-92c1-841b285cd101)
### h(x) Value :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/d8b272d5-104d-4cdb-942c-b849e8b54300)
### Cost function using Gradient Descent Graph :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/c1ac8e0b-f252-4aac-8984-2c9f00da624a)
### Profit for the Population 35,000 :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/61aad46d-b2d2-47d7-a7d7-ece05043cf30)
### Profit for the Population 70,000 :
![image](https://github.com/harini1006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/113497405/70f9f953-a1da-4225-be06-19b89e9b42fe)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
