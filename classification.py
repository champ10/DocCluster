import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import KFold

df_data = pd.read_csv("classify.csv")

Y =np.asarray( df_data.iloc[:,0])
X = np.asarray(df_data.iloc[:,1:18])

model = RandomForestClassifier(n_estimators=100,min_samples_leaf=2, min_samples_split=3,)
# scores = cross_val_score(model, X, Y, cv=10)
predicted = cross_val_predict(model, X, Y, cv=10)
print(metrics.f1_score(Y, predicted))

kf = KFold(n_splits=10)
f_score =[]
for traincv, testcv in kf.split(X):
   model.fit(X[traincv], Y[traincv])
   y_pred = model.predict(X[testcv])
   f_score.append(metrics.f1_score(Y[testcv], y_pred, average='micro'))

print ("Results: " + str( np.array(f_score).mean() ))