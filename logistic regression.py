


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
df=sns.load_dataset("titanic")
df.isnull().sum().sort_values(ascending=True)
df.drop("deck",axis=1,inplace=True)
df.drop("alive",axis=1,inplace=True)
df["age"]=df["age"].fillna(df["age"].median())
df["embark_town"]=df["embark_town"].fillna(df["embark_town"].mode())
df["embarked"]=df["embarked"].fillna(df["embarked"].mode())
for col in df.columns:
    if df[col].dtype=="object" or df[col].dtype=="category":
        df[col]=LabelEncoder().fit_transform(df[col])
x=df.drop("survived",axis=1)
y=df["survived"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
model=LogisticRegression()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)

print(classification_report(ytest,pred))
sca=classification_report(ytest,pred,output_dict=True)
scadf=pd.DataFrame(sca).transpose()
plt.figure(figsize=(7,7))
sns.heatmap(scadf.iloc[:-1,:-1],annot=True,cmap="viridis",fmt=".2f")
plt.show()
