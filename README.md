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
/*
Program to implement the linear regression using gradient descent.
Developed by: R.TEJASWINI
RegisterNumber: 212224230218
*/
```
        import numpy as np
        
        import pandas as pd
        
        from sklearn.preprocessing import StandardScaler
        
        def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
        
            X=np.c_[np.ones(len(X1)),X1]
            
            theta=np.zeros(X.shape[1]).reshape(-1,1)
            
            for _ in range(num_iters):
            
                predictions=(X).dot(theta).reshape(-1,1)
                
                errors=(predictions-y).reshape(-1,1)
                
                theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
                
                pass
                
            return theta
        
        
        data=pd.read_csv('/content/50_Startups.csv',header=None)
        
        print(data.head())
        
        
        X=(data.iloc[1:, :-2].values)
        
        print(X)
        
        
        X1=X.astype(float)
        
        scaler=StandardScaler()
        
        y=(data.iloc[1:,-1].values).reshape(-1,1)
        
        print(y)
        
        
        X1_Scaled=scaler.fit_transform(X1)
        
        Y1_Scaled=scaler.fit_transform(y)
        
        
        print(X1_Scaled)
        
        print(Y1_Scaled)
        
        theta=linear_regression(X1_Scaled,Y1_Scaled)
        
        new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
        
        new_Scaled=scaler.fit_transform(new_data)
        
        prediction=np.dot(np.append(1,new_Scaled),theta)
        
        prediction=prediction.reshape(-1,1)
        
        pre=scaler.inverse_transform(prediction)
        
        print(f"Predicted value: {pre}")
        
## Output:

Data Information

<img width="730" height="154" alt="Screenshot 2025-09-08 144812" src="https://github.com/user-attachments/assets/18fbfd52-0bea-47bb-97c9-d931e8e64e90" />

Value of X

<img width="399" height="827" alt="Screenshot 2025-09-08 144957" src="https://github.com/user-attachments/assets/4a5f5394-cc97-4e18-ac13-16fa26ce4849" />

Value of X1_Scaled

<img width="478" height="825" alt="image" src="https://github.com/user-attachments/assets/c3afe2ba-4f39-4d23-8b0c-fabaa92d4112" />

Predicted Value

<img width="379" height="40" alt="Screenshot 2025-09-08 145107" src="https://github.com/user-attachments/assets/ae5afa64-350c-45d5-b83a-b9b28a485396" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
