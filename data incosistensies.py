#####date format inconsistensies

import pandas as pd
data={
    "date":   ["2022-12-3", "2021-9-12", "13-8-2022"],
    "country":["USA",       "U.S.A",     "United State"],
    "name":   ["daniel",    "daneil",    "deniel"],
    "age f":  [50,           40,           70],
    "age s":  [70,           10,           17]
}
df=pd.DataFrame(data)
# x=df.head()
# print(x)

df["date"]=pd.to_datetime(df["date"],errors="coerce")   #--> is se ghalat date format ki jga  NaT ae ga
x=df.head()
print(x)


import pandas as pd
data={
    "date":   ["2022-12-3", "2021-09-12", "13-8-2022"],
    "country":["USA",       "U.S.A",     "United State"],
    "name":   ["daniel",    "daneil",    "deniel"],
    "age f":  [50,           40,           70],
    "age s":  [70,           10,           17]
}
df=pd.DataFrame(data)
def correct_date(date):
    try:
        return pd.to_datetime(date,dayfirst=True).strftime("%Y-%m-%d")       #--> is se hm apni dates k galat format ko 
    except:                                                                  #standard format me cinvert krte ha
        return None
    
df["date"]=df["date"].apply(correct_date)
x=df.head()
print(x)

import pandas as pd
data={
    "date":   ["2022-12-3", "2021-9-12", "13-8-2022"],
    "country":["USA",       "U.S.A",     "United State"],
    "name":   ["daniel",    "daneil",    "deniel"],
    "age f":  [50,           40,           70],
    "age s":  [70,           10,           17]
}
df=pd.DataFrame(data)
a={
    "usa":"USA",
    "U.S.A":"USA",         
    "America":"USA",
    "america":"USA",
    "United states":"USA",
     "United state":"USA",
      "United State":"USA"
}
x=df["country"]=df["country"].replace(a)  #-->isse hm replace krte ha data ko 1 standard format me wo country,name ho sakta ha
print(x)
