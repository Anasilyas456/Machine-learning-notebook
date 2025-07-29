import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

df=sns.load_dataset("tips")
x=df[["total_bill"]]
y=df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train,y_train)
print(model.intercept_)
print(model.coef_)
print("y=",model.intercept_,"+",model.coef_,"*x")
pred=model.predict(x_test)

print("mse =",mean_squared_error(y_test,pred))
print("r2 =",r2_score(y_test,pred))
plt.scatter(x_test,y_test)
plt.plot(x_test,pred,color="red")
plt.show()
