#DATA IS FROM: https://data.lacounty.gov/Criminal/LA-SHERIFF-CRIMES-FROM-2004-TO-2015/3dxh-c6jw
import csv
import numpy as np
import pandas as pd
import re
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw2/LA_CRIMESDATA_BACKUP.csv')
statistic = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw2/COUNT.csv')
#to see the contribution in every type of crimes
# for item in data:
# 	uniq = data[item].unique()
# 	print("'{}' has {} unique values".format(item,uniq.size))
# 	if(uniq.size>50):
# 		print "Only list 50 unique values:"
# 		print(uniq[0:49])
# 		print("\n---------------------------")

sns.set_style("whitegrid")
# to count the number of crimesevents in different crime category
print data.groupby(['CRIME_CATEGORY_NUMBER']).count()
sns.barplot(x='CRIME_CATEGORY',y='COUNT',data = statistic)
# plt.show()

# training_features = ['TIME_PERIOD','MONTH','DATE','CRIME_YEAR','ZIP','LATITUDE','LONGITUDE','EVENT']
# training_features = ['TIME_PERIOD','ZIP','LATITUDE','LONGITUDE','LONG_SQ','LATI_SQ']
training_features = ['TIME_PERIOD','ZIP','LATITUDE','LONGITUDE','LONG_SQ']
# training_features = ['ZIP']
X_train,X_test,y_train,y_test = cross_validation.train_test_split(data[training_features],data['CRIME_CATEGORY_NUMBER'],test_size=0.2,random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)
# param_grid = {"n_estimators":[10,25,50,100,200,500,1000],"max_features":[5],"min_samples_split":[100],"bootstrap":[True],"max_depth":[None],"criterion":["gini"]}
Alg = RandomForestClassifier(random_state=1,n_estimators=90,min_samples_split=8,max_depth=25)

# Alg = RandomForestClassifier(random_state=1,n_estimators=i,min_samples_split=100)
forest = Alg.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for x in range(X_train_trans.shape[1]):
	print("%2d) %-*s %f" % (x + 1, 30, 
                             training_features[indices[x]], 
                             importances[indices[x]]))	

train_score = forest.score(X_train,y_train)
test_score = forest.score(X_test,y_test)
print ("Train Score: %0.6f\nTest Score: %0.6f" %(train_score, test_score))

# forest = RandomForestClassifier()
# grid_search = GridSearchCV(forest,param_grid=param_grid)
# grid_search.fit(X_train_trans,y_train)
# print(grid_search.best_estimator_)
# train_score = grid_search.score(X_train_trans,y_train)
# test_score = grid_search.score(X_test_trans,y_test)
# print("Train Score: %0.4f\nTest Score:%0.4f" %(train_score,test_score))
