#!/usr/bin/env python
# coding: utf-8

# # Determine the success of an ad campaign.

# In[ ]:


# Determine the success of an ad campaign.
# The dataset we have contains a total of 12 columns and 26048 observations.
# So, based on these features and observations, we will predict if a particular ad will be successful in the future or not.
# We will visualize the data and try to understand the type of ad campaigns which are more successsful and focus on those kind 
# of ads in the future or try to understand why particular ads are more successful than others.
# With these kind of insights, we can also improve the unsuccessful ones.


# In[ ]:


#Data
#Data Description

#id -Unique id for each row

#ratings -Metric out of 1 which represents how much of the targeted demographic watched the advertisement

#airlocation -Country of origin

#airtime -Time when the advertisement was aired

#average_runtime(minutes_per_week) -Minutes per week the advertisement was aired

#targeted_sex -Sex that was mainly targeted for the advertisement

#genre -The type of advertisement

#industry -The industry to which the product belonged

#economic_status -The economic health during which the show aired

#relationship_status -The relationship status of the most responsive customers to the advertisement

#expensive -A general measure of how expensive the product or service is that the ad is discussing.

#money_back_guarantee -Whether or not the product offers a refund in the case of customer dissatisfaction.

#netgain [target] -Whether the ad will incur a gain or loss when sold


# In[47]:


#Import the libraries.

import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


# In[48]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_Interactivity = 'all'


# In[25]:


#Import the data
train=pd.read_csv('C:\\Users\\anju\\Desktop\\Data science misc\\datasets for practice\\Ad success/Train.csv')


# In[6]:


print('total train data: ' + str(train.shape))


# In[7]:


train.info()


# In[8]:


train.describe()


# In[9]:


train.isnull().sum()    # No missing values found.


# In[10]:


for i in train.columns:
    print( i , ':', train[i].nunique())
    


# In[11]:


train.head()


# In[26]:


#dropping id column as its of no use
train = train.drop(['id'],axis=1)
train.head()


# # EDA

# In[18]:


#Understanding the target variable netgain.
# We can see that the count of unsuccessful ads are higher.

sns.countplot('netgain',data = train) 


# In[167]:


# Around 76% of the ad campaigns are not successful.
# 0 - False = ad campaign not successful.
# 1 - True = ad campaign successful.

total = float(len(train))
plt.title('Netgain sucess and failure percentage')
ax = sns.countplot(x="netgain", data=train)
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[171]:


# 0 - false - blue colour - ad campaign not successful.
# 1 - true -  yellow colour - ad campaign successful.
# From below plot, we can infer that ads belonging to the 'comedy' genre are the most successful  amongst all the others. 
# So, Looks like people love to laugh and we can add a sigificant amount of comedy to almost every ad to make it a 
# successful one.
# However, even the comedy genre is successful only around 22% of the times.

plt.figure(figsize=(15,6))
plt.title('Netgain success based on genre')
total = float(len(train))
ax = sns.countplot(x="genre", data=train, hue = 'netgain')
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[172]:


# Male - blue colour 
# Female-  yellow colour 
# From below plot, we can infer that males watch more ads in all the genres.
# Since the percentage of male watching ads are more in all the genres , we can make the ads more specific and interesting 
# towards the male population to make them more successful.
# We can also see that around 58% of males watch comedy ads compared to 27% of females. Do men really like comedy so much? or
# it's just that women are more busy to waste time watching ads :-P
# Also comedy genre is more famous amongst the population irrespective of gender.Looks like we need to add more humour to our ads.


plt.figure(figsize=(15,6))
plt.title('Percentage of various Genres of commercials watched by different genders')
total = float(len(train))
ax = sns.countplot(x="genre", data=train, hue = 'targeted_sex')
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[173]:


# 0 - false - blue colour - ad campaign not successful.
# 1 - true -  yellow colour - ad campaign successful.
# From below plot, again its clear that men watch more ads.
# However only around 20% of the ads men watch are successful whereas around 4% of the ads female watch are succcessful.

plt.figure(figsize=(15,6))
plt.title('Netgain success based on gender')
total = float(len(train))
ax = sns.countplot(x="targeted_sex", data=train , hue = 'netgain')
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[174]:


# 0 - false - blue colour - ad campaign not successful.
# 1 - true -  yellow colour - ad campaign successful.
# From below plot, we can clearly say that the ads related to Pharma industry are more successful. It could also be because 
# Pharma industry dominates the other sectors. lets plot it and check it out.

plt.figure(figsize=(15,6))
total = float(len(train))
plt.title('netgain success based on industry type')
ax = sns.countplot(x="industry", data=train , hue = 'netgain')
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[175]:


# As suspected above, yes Pharma industry dominates the other sectors and has the highest count of more than 10000 observations
# realted to it.Around 40% of the industry sector is contributed to Pharma industry and it also has high count of successful ads.

plt.figure(figsize=(15,6))
plt.title('Industry Types')
total = float(len(train))
ax = sns.countplot(x="industry", data=train)
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[178]:


# 0 - false - blue colour - ad campaign not successful.
# 1 - true -  yellow colour - ad campaign successful.
# From below plot, we can clearly say that the success of the ads are more in the Married_civ_spouse category.

plt.figure(figsize=(15,6))
plt.title(' netgain success based on relationship_status')
total = float(len(train))
ax = sns.countplot(x="realtionship_status", data=train , hue = 'netgain')
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[186]:


# 0 - false - Light blue colour - ad campaign not successful.
# 1 - true - pink colour - ad campaign successful.
# The low expensive ads are the most successful ones.

plt.figure(figsize=(15,6))
plt.title(' netgain success based on expense rate')
total = float(len(train))
ax = sns.countplot(x="expensive", data=train , hue = 'netgain' , palette="Set2")
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[209]:


# 0 - false - red colour - ad campaign not successful.
# 1 - true -  blue colour - ad campaign successful.

# The ads aired during primetime are the most successful ones.
# Hence, we need to air more ads during primetime.

plt.figure(figsize=(15,6))
plt.title(' netgain success based on aired time')
total = float(len(train))
ax = sns.countplot(x="airtime", data=train , hue = 'netgain' , palette="Set1")
for i in ax.patches:
    height = i.get_height()
    ax.text(i.get_x()+i.get_width()/2.,height + 5,
           '{:1.2f}'.format(height/total*100) + '%')


# In[188]:


# We can see that the average run time of the ads per week is around 40 mins i.e the ads were aired around 40 mins per week.

plt.figure(figsize=(15,6))
sns.distplot(train['average_runtime(minutes_per_week)'])
plt.show()


# In[189]:


plt.figure(figsize=(25,20))
sns.factorplot(data=train,x='netgain',y='ratings',hue='genre')


# In[190]:


# Daytime ads are run more amount of time compared to the other airtimes. 

sns.catplot(x='airtime', y='average_runtime(minutes_per_week)', data=train, kind='boxen', aspect=2)
plt.title('Boxen Plot', weight='bold', fontsize=16)
plt.show()


# In[198]:


# Ads from pharma industry are aired more compared to others.

plt.figure(figsize=(200,400))
sns.factorplot(data=train,x='industry',y='average_runtime(minutes_per_week)')
plt.title('Factor Plot', weight='bold', fontsize=16)
plt.show()


# In[199]:


# High expensive ads got the highest ratings.

sns.catplot(x='expensive', y='ratings', data=train, kind='boxen', aspect=2)
plt.title('Boxen Plot', weight='bold', fontsize=16)
plt.show()


# In[29]:


# Since my target variable 'netgain' is boolean , I will use label encoding for the same

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(train['netgain'])
print(list(le.classes_))
train['netgain'] = le.transform(train['netgain'])
train.head()


# In[28]:


# My dataset has 8 categorical columns which we need to encode. Here I am using one hot encoding.

train_encode = pd.get_dummies(train)


# In[30]:


train_encode.shape


# In[33]:


# Splitting the independent and target variables.

train_ind = train_encode.drop(['netgain'],axis=1)
target = train_encode['netgain']


# In[18]:


print( 'Train independent dataset shape:', train_ind.shape , 'and', 'Train target dataset shape:', target.shape)


# In[34]:


#Splitting the dataset to train and test.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_ind, target, test_size=0.30, random_state=42)

print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)


# # Decision tree

# In[104]:


# Decision tree classifier with grid seacrh CV and model evaluation using accuracy score, precision score and AUC/ROC curve.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

parameters = {'max_features': [0.5,0.6,0.7,0.8,0.9,1.0], 'max_depth': [2,3,4,5,6,7],'min_samples_leaf':[1,10,100],'random_state':[14]} 

clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, scoring='roc_auc')
clf.fit(X_train, y_train)


# In[105]:


print('The best parameters are: ', clf.best_params_)
print('best mean cross-validated score (auc) : ', clf.best_score_)


# In[110]:


from sklearn.metrics import precision_score, accuracy_score
x_actual, x_pred = y_train, clf.predict(X_train)
precision_score_DT_train = precision_score(x_actual, x_pred)
accuracy_score_DT_train = accuracy_score(x_actual, x_pred)
print('The precision score of decision tree on TRAIN is : ',round(precision_score_DT_train * 100,2), '%')
print('The accuracy score of decision tree on TRAIN is : ',round(accuracy_score_DT_train * 100,2), '%')


# In[111]:


from sklearn.metrics import precision_score, accuracy_score
y_actual, y_pred = y_test, clf.predict(X_test)
precision_score_DT_test =  precision_score(y_actual, y_pred)
accuracy_score_DT_test = accuracy_score(y_actual, y_pred)
print('The precision score of decision tree on TEST is : ',round(precision_score_DT_test * 100,2), '%')
print('The accuracy score of decision tree on TEST is : ',round(accuracy_score_DT_test * 100,2), '%')


# In[ ]:


# Among all predictions for ad to be successful, 65.8% are successful.
# 81.5 % of the succcessful ads were predicted correctly.


# In[113]:


#Now let's plot the ROC curve and calculate AUC on the test set
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
adsu = clf.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

AUC_DT = auc(fpr,tpr)
print('DT AUC is: ', round(AUC_DT * 100,2), '%')


# In[51]:


#getting the confusion matrix for the classification model
from sklearn.metrics import confusion_matrix # cofusion matrix / accuracy
print ('Confusion Matrix TRAIN:\n', confusion_matrix(y_train,x_pred))
print ('\nConfusion Matrix TEST:\n', confusion_matrix(y_test,y_pred))


# In[54]:


# getting the classification report of the classification models
from sklearn.metrics import classification_report 
print ('Classification Report TRAIN:\n', classification_report(y_train,x_pred))
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))


# # Random Forest

# In[114]:



from sklearn.ensemble import RandomForestClassifier
parameters = {'n_estimators':[700],'n_jobs':[-1], 'max_features': [0.5,0.7,0.9], 'max_depth': [3,5,7],'min_samples_leaf':[1,10],'random_state':[14]} 

clf1 = GridSearchCV(RandomForestClassifier(), parameters, cv=5, scoring='roc_auc')
clf1.fit(X_train, y_train)


# In[115]:


print('The best parameters are: ', clf1.best_params_)
print('best mean cross-validated score (auc) : ', clf1.best_score_)


# In[116]:


from sklearn.metrics import precision_score, accuracy_score
x_actual, x_pred = y_train, clf1.predict(X_train)
precision_score_RF_train = precision_score(x_actual, x_pred)
accuracy_score_RF_train = accuracy_score(x_actual, x_pred)
print('The precision score of random forest on TRAIN is : ',round(precision_score_RF_train * 100,2), '%')
print('The accuracy score of random forest on TRAIN is : ',round(accuracy_score_RF_train * 100,2), '%')


# In[117]:


from sklearn.metrics import precision_score, accuracy_score
y_actual, y_pred = y_test, clf1.predict(X_test)
precision_score_RF_test =  precision_score(y_actual, y_pred)
accuracy_score_RF_test = accuracy_score(y_actual, y_pred)
print('The precision score of random forest on TEST is : ',round(precision_score_RF_test * 100,2), '%')
print('The accuracy score of random forest on TEST is : ',round(accuracy_score_RF_test * 100,2), '%')


# In[ ]:


# Among all predictions for ad to be successful, 65.8% are successful.
# 81.5 % of the succcessful ads were predicted correctly.


# In[118]:


#Now let's plot the ROC curve and calculate AUC on the test set
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
adsu = clf1.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

AUC_RF = auc(fpr,tpr)
print('RF AUC is: ', round(AUC_RF * 100,2), '%')


# In[68]:


#getting the confusion matrix for the classification model
from sklearn.metrics import confusion_matrix # cofusion matrix / accuracy
print ('Confusion Matrix TRAIN:\n', confusion_matrix(y_train,x_pred))
print ('\nConfusion Matrix TEST:\n', confusion_matrix(y_test,y_pred))


# In[69]:


# getting the classification report of the classification models
from sklearn.metrics import classification_report 
print ('Classification Report TRAIN:\n', classification_report(y_train,x_pred))
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))


# # Voting Classifier

# In[119]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

rfClf = RandomForestClassifier(n_estimators=500, random_state=0) # 500 trees. 
svmClf = SVC(probability=True, random_state=0) # probability calculation
logClf = LogisticRegression(random_state=0)
#nbclf = GaussianNB(random_state=0)

# constructing the ensemble classifier by mentioning the individual classifiers.
clf2 = VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf), ('log', logClf)], voting='soft') 

# train the ensemble classifier
clf2.fit(X_train, y_train) 


# In[120]:


from sklearn.metrics import precision_score, accuracy_score
x_actual, x_pred = y_train, clf2.predict(X_train)
precision_score_VC_train = precision_score(x_actual, x_pred)
accuracy_score_VC_train = accuracy_score(x_actual, x_pred)
print('The precision score of Voting classifier on TRAIN is : ',round(precision_score_VC_train * 100,2), '%')
print('The accuracy score of Voting classifier on TRAIN is : ',round(accuracy_score_VC_train * 100,2), '%')


# In[121]:


from sklearn.metrics import precision_score, accuracy_score
y_actual, y_pred = y_test, clf2.predict(X_test)
precision_score_VC_test = precision_score(y_actual, y_pred)
accuracy_score_VC_test = accuracy_score(y_actual, y_pred)
print('The precision score of Voting classifier on Test is : ',round(precision_score_VC_test * 100,2), '%')
print('The accuracy score of Voting classifier on Test is : ',round(accuracy_score_VC_test * 100,2), '%')


# In[122]:


#Now let's plot the ROC curve and calculate AUC on the test set
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
adsu = clf2.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

AUC_VC = auc(fpr,tpr)
print('VC AUC is: ', round(AUC_VC * 100,2), '%')


# In[90]:


#getting the confusion matrix for the classification model
from sklearn.metrics import confusion_matrix # cofusion matrix / accuracy
print ('Confusion Matrix TRAIN:\n', confusion_matrix(y_train,x_pred))
print ('\nConfusion Matrix TEST:\n', confusion_matrix(y_test,y_pred))


# In[91]:


# getting the classification report of the classification models
from sklearn.metrics import classification_report 
print ('Classification Report TRAIN:\n', classification_report(y_train,x_pred))
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))


# # Boosting algorithm - XGBoost

# In[123]:


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05)
    
xgb_clf.fit(X_train,y_train)
    
    


# In[124]:


from sklearn.metrics import precision_score, accuracy_score
x_actual, x_pred = y_train, xgb_clf.predict(X_train)
precision_score_XG_train = precision_score(x_actual, x_pred)
accuracy_score_XG_train = accuracy_score(x_actual, x_pred)
print('The precision score of XGBOOST on TRAIN is : ',round(precision_score_XG_train * 100,2), '%')
print('The accuracy score of XGBOOST on TRAIN is : ',round(accuracy_score_XG_train * 100,2), '%')


# In[125]:


from sklearn.metrics import precision_score, accuracy_score
y_actual, y_pred = y_test, xgb_clf.predict(X_test)
precision_score_XG_test = precision_score(y_actual, y_pred)
accuracy_score_XG_test = accuracy_score(y_actual, y_pred)
print('The precision score of XGBOOST on Test is : ',round(precision_score_XG_test * 100,2), '%')
print('The accuracy score of XGBOOST on Test is : ',round(accuracy_score_XG_test * 100,2), '%')


# In[126]:


#Now let's plot the ROC curve and calculate AUC on the test set
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
adsu = xgb_clf.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

AUC_XG = auc(fpr,tpr)
print('XG AUC is: ', round(AUC_XG * 100,2), '%')


# In[99]:


#getting the confusion matrix for the classification model
from sklearn.metrics import confusion_matrix # cofusion matrix / accuracy
print ('Confusion Matrix TRAIN:\n', confusion_matrix(y_train,x_pred))
print ('\nConfusion Matrix TEST:\n', confusion_matrix(y_test,y_pred))


# In[100]:


# getting the classification report of the classification models
from sklearn.metrics import classification_report 
print ('Classification Report TRAIN:\n', classification_report(y_train,x_pred))
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))


# # Reporting the accuracy of all the data models used for prediction.

# In[151]:


print('The ACCURACY SCORE of various models on train and test are:')
print('                                                         ')
print('The accuracy score of decision tree on : \t\t\t\t\t\tTRAIN:',round(accuracy_score_DT_train * 100,2), '\b%','\tTEST:',round(accuracy_score_DT_test * 100,2), '\b%')
print('The accuracy score of random forest ensembler on: \t\t\t\t\tTRAIN:',round(accuracy_score_RF_train * 100,2), '\b%','\tTEST:',round(accuracy_score_RF_test * 100,2), '\b%')
print('The accuracy score of multiple different classifiers - Voting classifier on: \t\tTRAIN:',round(accuracy_score_VC_train * 100,2), '\b%','\tTEST:',round(accuracy_score_VC_test * 100,2), '\b%')
print('The accuracy score of boosting model XGBOOST on: \t\t\t\t\tTRAIN:',round(accuracy_score_XG_train * 100,2), '\b%','\tTEST:',round(accuracy_score_XG_test * 100,2), '\b%')


# In[148]:


print('The PRECISION SCORE of various models on train and test are:')
print('                                                         ')
print('The precision score of decision tree on : \t\t\t\t\t\tTRAIN:',round(precision_score_DT_train * 100,2), '\b%','\tTEST:',round(precision_score_DT_test * 100,2), '\b%')
print('The precision score of random forest ensembler on: \t\t\t\t\tTRAIN:',round(precision_score_RF_train * 100,2), '\b%','\tTEST:',round(precision_score_RF_test * 100,2), '\b%')
print('The precision score of multiple different classifiers - Voting classifier on: \t\tTRAIN:',round(precision_score_VC_train * 100,2), '\b%','\tTEST:',round(precision_score_VC_test * 100,2), '\b%')
print('The precision score of boosting model XGBOOST on: \t\t\t\t\tTRAIN:',round(precision_score_XG_train * 100,2), '\b%','\tTEST:',round(precision_score_XG_test * 100,2), '\b%')


# In[155]:


print('The AUC SCORE of various models on test are:')
print('                                                         ')
print('The AUC score of decision tree on : \t\t\t\t\t\t\tTest:',round(AUC_DT * 100,2), '%', '\b%')
print('The AUC score of random forest ensembler on: \t\t\t\t\t\tTest:',round(AUC_RF * 100,2), '%', '\b%')
print('The AUC score of multiple different classifiers - Voting classifier on: \t\tTest:',round(AUC_VC * 100,2), '%', '\b%')
print('The AUC score of boosting model XGBOOST on: \t\t\t\t\t\tTest:',round(AUC_XG * 100,2), '%', '\b%')


# In[ ]:


# Hence, we can say that boosting algorithm XGBoost performed the best on our dataset. It will predict most precisely the 
# success of the ad campaign.


# # Single SVM

# In[87]:


# Single SVM
from sklearn.svm import SVC
svmClf = SVC(probability=True)
svmClf.fit(X_train, y_train)

y_pred = svmClf.predict(X_test)


# In[79]:


print('precision of single SVM on the test set: ', round(precision_score(y_test, y_pred)*100,2),'%')
print('accuracy of single SVM on the test set: ', round(accuracy_score(y_test, y_pred)*100,2),'%')


# In[80]:


svmprob = svmClf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, svmprob)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('AUC for single SVM is: ', round(auc(fpr,tpr)*100,2),'%')


# In[88]:


#getting the confusion matrix for the classification model
from sklearn.metrics import confusion_matrix # cofusion matrix / accuracy
print ('\nConfusion Matrix of single SVM TEST:\n', confusion_matrix(y_test,y_pred))


# In[89]:


from sklearn.metrics import classification_report 
print ('\nClassification Report of single SVM TEST:\n', classification_report(y_test,y_pred))


# # Ensemble of multiple SVM using bagging classifier.

# In[81]:


#Ensemble of multiple SVM using bagging classifier.

from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

svmClfbag = BaggingClassifier(SVC(C=1.0, kernel='linear', degree=5, gamma='auto', coef0=0.0, shrinking=True, probability=True,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))
svmClfbag.fit(X_train, y_train)

#print(svmClfbag.oob_score_)

y_pred = svmClfbag.predict(X_test)


# In[82]:


print('precision of multiple SVM  on the test set: ', round(precision_score(y_test, y_pred)*100,2),'%')
print('accuracy of multiple SVM on the test set: ', round(accuracy_score(y_test, y_pred)*100,2),'%')


# In[83]:


svmprobbag = svmClf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, svmprobbag)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('AUC for multiple SVM - Bagging Classifier is: ', round(auc(fpr,tpr)*100,2),'%')


# In[85]:


#getting the confusion matrix for the classification model
from sklearn.metrics import confusion_matrix # cofusion matrix / accuracy
print ('\nConfusion Matrix TEST:\n', confusion_matrix(y_test,y_pred))


# In[86]:


from sklearn.metrics import classification_report 
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))


# # Single Logistic Regression Classifier
# 

# In[62]:


#Single Logistic Regression

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=0, solver='lbfgs') 
log.fit(X_train, y_train)

y_pred = log.predict(X_test)


# In[64]:


print('precision of single logistic regression classifier on the test set: ', round(precision_score(y_test, y_pred)*100,2),'%')
print('accuracy of single logistic regression classifier on the test set: ', round(accuracy_score(y_test, y_pred)*100,2),'%')


# In[65]:


logauc = log.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, logauc)

plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('single Logistic regression AUC is: ', round(auc(fpr,tpr)*100,2),'%')


# # Multiple logistic regression classifiers using bagging Classifier

# In[66]:


# Multiple logistic regression classifiers using bagging Claissifier.
logbagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 400, oob_score = True, random_state = 90)
logbagClf.fit(X_train, y_train)


# In[76]:


print('OOB score of an ensemble of multiple logistic regression classifiers:', round((logbagClf.oob_score_)*100,2),'%')

# The oob score is an actual estimate of the accuracy of the ensemble classifier 


# In[69]:



y_pred = logbagClf.predict(X_test)
print('precision of an ensemble of multiple logistic regression classifiers on the test set: ', round(precision_score(y_test, y_pred)*100,2),'%')
print('accuracy of an ensemble of multiple logistic regression classifiers on the test set: ', round(accuracy_score(y_test, y_pred)*100,2),'%')


# In[71]:


#AUC score
logbag = logbagClf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, logbag)
plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

print('AUC of baggingclassifier of logistic regression classifiers is: ', round(auc(fpr,tpr)*100,2),'%')

