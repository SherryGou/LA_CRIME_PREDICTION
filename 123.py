import csv
import numpy as np
import pandas as pd
import re
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# import seaborn as sns

data = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw2/LA_CRIMESDATA_BACKUP.csv')

statistic = pd.read_csv('/Users/xuanxuan/Documents/inf552/hw2/COUNT.csv')

# to count the number of crimesevents in different crime category
print data.groupby(['CRIME_CATEGORY_NUMBER']).count()
# sns.barplot(x='CRIME_CATEGORY',y='COUNT',data = statistic)
# plt.show()

# training_features = ['TIME_PERIOD','MONTH','DATE','CRIME_YEAR','ZIP','LATITUDE','LONGITUDE','EVENT']
training_features = ['TIME_PERIOD','ZIP','LATITUDE','LONGITUDE','LONG_SQ','LATI_SQ']
# training_features = ['ZIP']
X_train,X_test,y_train,y_test = cross_validation.train_test_split(data[training_features],data['CRIME_CATEGORY_NUMBER'],test_size=0.2,random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)
#to find the correlation between number of trees and train score/test score
f = open('treefile','w')
for i in range(50,201,10):
	Alg = RandomForestClassifier(random_state=1,n_estimators=i,min_samples_split=8)
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
	f.write("%d %0.6f %0.6f\n" %(i, train_score, test_score))
f.close

x = open('treefile','r')

lineList = x.readlines()
lineList = [line.strip().split(' ') for line in lineList]
x.close
no_of_trees = [x[0] for x in lineList]
train_score1 = [x[1] for x in lineList]
test_score1 = [x[2] for x in lineList]
plt.plot(no_of_trees, train_score1, label = 'Train_score', color = 'red')
plt.plot(no_of_trees, test_score1, label = 'Test_score', color = 'blue')
plt.xlabel('No_of_Trees')
plt.ylabel('Score')
plt.title('Model accuracy vs. no_of_trees')
# plt.show()
plt.savefig('accuracy_vs_no_of_trees.png')

