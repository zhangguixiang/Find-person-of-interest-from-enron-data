#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
### Remove data point 'TOTAL'
data_dict.pop('TOTAL',0)
data_dict.pop('TRAVEL AGENCY IN THE PARK',0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

###Extrat email features first
features_list=['poi','to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi']

### Extract features and labels from dataset for local testing

### To rewrite in my_data, set sort_key=False, remove_all_zeroes=False
data = featureFormat(my_dataset, features_list, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)

###calculate 
###calculate from poi fraction, to poi fraction and shared with poi fraction
features_fraction=[]
import numpy
for feature in features:
    feature_fraction=numpy.array([0.0,0.0,0.0])
    if feature[0] == 0:
        feature_fraction[0]=0
        feature_fraction[2]=0
    else:
        feature_fraction[0]=feature[1]/feature[0]
        feature_fraction[2]=feature[4]/feature[0]
    if feature[2] == 0:
        feature_fraction[1]=0
    else:
        feature_fraction[1]=feature[3]/feature[2]
    features_fraction.append(feature_fraction)

###select rows where not all items are zero
features_not_all_zero=[]
dopca_index=[]
for i,row in enumerate(features_fraction):
    append = False
    for item in row:
        if item !=0:
            append = True
            break
    if append:
        features_not_all_zero.append(row)
        dopca_index.append(i)
  
###do PCA for the three poi related fraction features and write the result to the numpy array rows where not all poi items equal to zero
from sklearn.decomposition import PCA
pca=PCA(n_components=1)
poi_index=numpy.zeros((len(data),1),dtype=float)
poi_index[[dopca_index]]=pca.fit_transform(features_not_all_zero)

###write poi_index back to my_dataset
keys=my_dataset.keys()
i=0
for key in keys:
    my_dataset[key]['poi_index']=poi_index[i][0]
    i=i+1

### Extract finance features and labels from dataset for local testing
features_list=['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
### select 3 most powerful finance features by SelectKbest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selector=SelectKBest(f_classif,k=4)
selector.fit(features,labels)
### score of each finance feature
score=selector.scores_
finance_index=numpy.argsort(-score)
### select 4 most powerful finance features
finance_feature_select=[]
features_list.remove('poi')
for i in finance_index[:4]:
    finance_feature_select.append(features_list[i])

###do min-max scaling
###select features
finance_feature_select.insert(0,'poi')
features_list = finance_feature_select
features_list.append('poi_index')
data = featureFormat(my_dataset, features_list, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)
#import min-max scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features)
#write back to my_dataset
keys=my_dataset.keys()
i=0
for key in keys:
    my_dataset[key]['exercised_stock_options']=features[i][0]
    my_dataset[key]['total_stock_value']=features[i][1]
    my_dataset[key]['bonus']=features[i][2]
    my_dataset[key]['salary']=features[i][3]
    my_dataset[key]['poi_index']=features[i][4]
    i=i+1

#features_list=['poi','total_payments', 'restricted_stock', 'deferral_payments']
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#Try and test naive_bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#Try and test decision tree
#from sklearn.tree import DecisionTreeClassifier
#clf=DecisionTreeClassifier()

#Try and test adaboost
#from sklearn.ensemble import AdaBoostClassifier
#clf=AdaBoostClassifier()

#Try and test SVM
#from sklearn.svm import SVC
#clf=SVC(kernel='linear')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#Use GridSearchCV to tune parameters for DecisionTreeClassifier
#from sklearn.model_selection import GridSearchCV
#param_grid = {'min_samples_split': [2, 5, 10, 15]}
#clf = GridSearchCV(DecisionTreeClassifier(), param_grid)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#fit classifier by training data set
clf.fit(features_train,labels_train)
#do prediction by testing training set
pred=clf.predict(features_test)
#test precision and recall with testing label set to evaluate performance
from sklearn.metrics import precision_score, recall_score
precision=precision_score(pred,labels_test)
recall=recall_score(pred,labels_test)
print "Precision of the algorithm is",precision
print "Recall of the algorithm is",recall


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)