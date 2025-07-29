#isko hm isliye use krte ha tak hamara alphabet data ko number me convert krske jo machine ko
#samajhne me asani paida krta ha or time km leta ha respons krne me.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=sns.load_dataset("tips")
df=pd.DataFrame(data)
x=df["day"].value_counts()
print(df.head())
print(x)
#ye labelencoder k liye use ho rha ha.ye meri coloumn ki value ko numbers me convert krde ga.
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
le=LabelEncoder()
df["dle"]=le.fit_transform(df["sex"])
print(df.head())
a=df["dle"].value_counts()
print(a)

#####ye ordinalencoder k liye ha
from sklearn.preprocessing import OrdinalEncoder
oe=OrdinalEncoder(categories=[["Thur","Fri","Sat","Sun"]])  #-->is line me days k name hme show kr rhe ha k hmne order pr prior kisko rkha ha.
df["doe"]=oe.fit_transform(df[["day"]])
print(df.head())
a=df["doe"].value_counts()
print(a)
