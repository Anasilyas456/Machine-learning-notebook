#finding missing data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=sns.load_dataset("titanic")
plt.figure(figsize=(8,5))
sns.heatmap(df.isnull(),cbar=False)                   #bar ki format me show krwane k liye ye line likhi ha sirf khud k analysis k liye
plt.show()

x=df.isnull().sum().sort_values(ascending=False)      # 2 method data ko full format me find krne ka
print(x)

#imputing missing values
df["age"]=df["age"].fillna(df["age"].mean())                     #1 method via pandas  (mean ki jga median or mode bhi use kr sakte ha)
df.drop("age",axis=1,inplace=True)               #----->           agar kisi coloumn ko del krna ho to hm isko use krte ha


from sklearn.impute import SimpleImputer                          # 2 method via scikit   ye sirf 1 coloumn me hi imputation k liye use hoga
imputer=SimpleImputer(strategy="median")
df["age"]=imputer.fit_transform(df[["age"]])

from sklearn.experimental import enable_iterative_imputer         #3 method via scikit (ye zyada accurate hota ha q k is se hm usk 
from sklearn.impute import IterativeImputer                        #  neighbour data ki basis pr impute krte ha values ko)
imputer=IterativeImputer(max_iter=20,n_nearest_features=8)  #------> is se hm usk neighbour se data fetch krte ha
df["age"]=imputer.fit_transform(df[["age"]])
x=df.isnull().sum().sort_values(ascending=False)          # ye line overalll data se imputing value show krwae gi
print(x)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=sns.load_dataset("titanic")
df=pd.DataFrame(data)
x=df.isnull().sum().sort_values(ascending=False)
x.head()
print(x)


print("impute data")
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
df["age"]=imputer.fit_transform(df[["age"]])
df.drop("deck",axis=1,inplace=True)
x=df.isnull().sum().sort_values(ascending=False)
print(x)
