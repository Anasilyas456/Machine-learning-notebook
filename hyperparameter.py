##ye hamare pass wo parameters hote ha jo hm model training k doran model ko assign krte ha tak best output ae.
##yha hm best parameter ko nikalne k liye method dekte ha

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
iris=load_iris()
x=iris.data
y=iris.target
model=RandomForestClassifier()
##now we create a parameter grid
grid_par={
    "n_estimators":[50,100,200,300,400,500],
    "max_features":["sqrt","log2"],
    "criterion":["gini","entropy","log_loss"],
    "max_depth":[2,3,4,5,6,7,8,9]
}

##set up a grid
grid=GridSearchCV(
    estimator=model,
    param_grid=grid_par,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1

)
grid.fit(x,y)
print(grid.best_params_)
