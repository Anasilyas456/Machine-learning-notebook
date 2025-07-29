   #isse hm apne data ki continous numerical values ko categorical values me convert krte ha.include 2 methods.
   #via scikit learn

  #import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
#load data
df=sns.load_dataset("titanic")
#imputing missing values
df["age"]=df["age"].fillna(df["age"].median())
#now we can discretize the data
label=["child","young","teenager","old","elder"]
disc=KBinsDiscretizer(n_bins=5,encode="ordinal",strategy="uniform")
df["age-d"]=disc.fit_transform(df[["age"]])
print(df)
print(df["age-d"].value_counts())
sns.histplot(df,x="age",hue="age-d")
plt.show()

   #via pandas
   #import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
   #load data
df=sns.load_dataset("titanic")
#imputing missing values
df["age"]=df["age"].fillna(df["age"].median())
#now we can discretize the data
label=[1,10,25,65]
df1=pd.cut(df["age"],bins=4,labels=label)
print(df1)
sns.histplot(df,x="age",hue=df1)
plt.show()

   #hm is chez ko manually bhi kr sakte ha
bins=[0,10,20,30,45,100]
label=["children","teenager","young","old","elder"]
df2=pd.cut(df["age"],bins=bins,labels=label)
print(df2)
sns.histplot(df,x="age",hue=df2)
plt.show()
