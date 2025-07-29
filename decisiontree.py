import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

df=sns.load_dataset("titanic")

df.drop("deck",axis=1,inplace=True)
df.drop("alive",axis=1,inplace=True)
imputer=SimpleImputer(strategy="median")
df[["age","fare"]]=imputer.fit_transform(df[["age","fare"]])
for col in df.columns:
    if df[col].dtype=="object"or df[col].dtype=="category":
        df[col]=LabelEncoder().fit_transform(df[col])
x=df.drop("survived",axis=1)
y=df["survived"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(criterion="entropy",max_depth=3)
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(classification_report(ytest,pred))
