


#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer

#import warnings
import warnings
warnings.filterwarnings('ignore')
#load data
data=sns.load_dataset('titanic')
# ## print(data.head())
# ## print(data.isnull().sum().sort_values(ascending=False))

#cleaning the data
df=data.drop(['age','deck','alive'],axis=1)

df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town']=df['embark_town'].fillna(df['embark_town'].mode()[0])
#print(df.isnull().sum().sort_values(ascending=False))

#encoding the data
encoder=LabelEncoder()
for col in df.columns:
    if df[col].dtypes=='object' or df[col].dtypes.name=='category':
        df[col]=encoder.fit_transform(df[col])

#train the data
x=df.drop('survived',axis=1)
y=df['survived']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

#scale the data
scaler=StandardScaler()
xtscaled=scaler.fit_transform(xtrain)
xscaled=scaler.transform(xtest) 

#make a neural network
il=tf.keras.layers.Dense(10,activation='relu',input_shape=(xtscaled.shape[1],))
ol=tf.keras.layers.Dense(1,activation='sigmoid')
mod=tf.keras.models.Sequential([il,ol])
mod.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
mod.fit(xtscaled,ytrain,epochs=100,batch_size=32,verbose=1)
accuracy=mod.evaluate(xscaled,ytest,verbose=1)
print(accuracy)
