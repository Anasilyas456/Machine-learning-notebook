import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df=sns.load_dataset("diamonds")

for col in df.columns:
    if df[col].dtype=="object"or df[col].dtype=="category":
        df[col]=LabelEncoder().fit_transform(df[col])
x=df.drop("cut",axis=1)
y=df["cut"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(criterion="entropy",max_depth=20)
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(classification_report(ytest,pred))
