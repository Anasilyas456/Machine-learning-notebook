  #import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
  #load data
df=sns.load_dataset("tips")
print(df.head())
 #separate features in X(jis pr predi depend kre) and label in y(jiski value pred krni ho)
X=df[["total_bill"]]
y=df["tip"]
 #train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
  # now we preprocess the data.
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
  # call and train the model
model=LinearRegression()
model.fit(X_train,y_train)
 #predict the value
xa=model.predict(X_test)
print(xa)

  ##real example


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df=sns.load_dataset("tips")
x=df[["total_bill"]]
y=df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train,y_train)
enter=float(input("enter your total amount of bill in $ :"))
a=scaler.transform(pd.DataFrame([[enter]],columns=["total_bill"]))
b=model.predict(a)
print(f"your predicted tip is:${b[0]:.2f}")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=sns.load_dataset("tips")
print(df.head())
x=df[["total_bill"]]
y=df["tip"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=LinearRegression()
model.fit(x_train,y_train)
enter=int(input("enter a total amount of bill in $:"))
a=scaler.transform(pd.DataFrame([[enter]],columns=["total_bill"]))
b=model.predict(a)
print(f"your predicted tip is:${b[0]:.2f}")
