
# coding: utf-8

# # Import Dataset and Pre-processing

# In[1]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
np.random.seed(0)

#Loading data
train = pd.read_csv('train_users_2.csv', encoding = "ISO-8859-1")
test = pd.read_csv('test_users.csv', encoding = "ISO-8859-1")
labels = train['country_destination'].values

# tem2 = train[(train.country_destination == 'NDF')].loc[::2]  # check reduce unbalanced class varibale samples
# balancetrain = train.merge(tem2, how='left', indicator=True)
# balancetrain = newtrain[balancetrain._merge == 'left_only']

# balancelabels = balancetrain['country_destination'].values
# # balancetrain = balancetrain.ix[:,0:-1]
print train.shape, # balancetrain.shape


# In[2]:

train = train.drop(['country_destination'], axis=1)
# balancetrain = balancetrain.drop(['country_destination'], axis=1)

train = train.drop(['id'],axis=1)
# balancetrain = balancetrain.drop(['id'],axis=1)

testid = test['id'].values
test = test.drop(['id'],axis=1)

print("Training Data shape:\n",train.shape )
# print("Balance_train Data shape:\n", balancetrain.shape)
print("Testing Data shape:\n",test.shape )
print("Target Data shape:\n",labels.shape )
# print("Balance Target Data shape:\n", balancelabels.shape)

print("Training Data: \n", train)
# print("Training Data: \n", balancetrain)
print("Testing Data: \n", test)
print("Target Variable: \n",labels)
# print("Target Variable: \n",balancelabels)


# ## Preprocessing on Training and Testing 
# 

# In[3]:

# Training + Testing
full = pd.concat((train, test), axis=0, ignore_index=True)
# balancefull = pd.concat((balancetrain, test), axis=0, ignore_index=True)
print(full.shape)  # 275547 * 14
# print(balancefull.shape)

headers= (list(full))  # All feature Titles


# In[4]:

# Check If Missing Value Involved in Full dataset 
full.isnull().sum()


# In[74]:

# Check If Missing Value Involved in Full dataset 
# balancefull.isnull().sum()


# ## Delete columns with over 60 percent missings

# In[5]:

# Delete columns with over 60 percent missings
# print(full.shape)  275547 * 14
full.dropna(thresh = 14*0.6, inplace = 'True')
print(full.shape)
# The reason we do not delete samples with missing is because in the tesing dataset,\
# there might exist missing as well, and we have to predict based on the given orignal test data.

# Result: No features dropped. 


# ## Fill Missing values

# In[77]:

## Fill Missing values in balancefull
# Fill missing age by average age
# balancefull['age'].fillna(balancefull['age'].mean(), inplace=True)
# balancefull['age'] = np.where(np.logical_or(balancefull.age.values<14, balancefull.age.values>90), balancefull['age'], balancefull.age.values)

# balancefull['first_affiliate_tracked'].fillna('NaN',inplace=True)
# balancefull['date_first_booking'].fillna('0-0-0',inplace=True)
# balancefull.isnull().sum()


# In[6]:

## Fill Missing values in full
# Fill missing age and outlier age by average age
full['age'].fillna(full['age'].mean(), inplace=True)
full['age'] = np.where(np.logical_or(full.age.values<14, full.age.values>90), full['age'], full.age.values)

full['first_affiliate_tracked'].fillna('NaN',inplace=True)
full['date_first_booking'].fillna('0-0-0',inplace=True)
full.isnull().sum()


# ## Feature LabelEncoder

# In[7]:

from sklearn import preprocessing
converting_list = [3,5,7,8,9,10,11,12,13]
namelist = list(full)
for ele in converting_list:
    if ele == converting_list[0]:
        col = full.ix[:,ele:ele+1]
        col = np.array(col)
        col = [val for sublist in col for val in sublist]
        col = np.array(col)
        le = LabelEncoder()
        crt = le.fit_transform(col)
        data = pd.Series(crt, name= namelist[ele])
    else:
        col = full.ix[:,ele:ele+1]
        col = np.array(col)
        col = [val for sublist in col for val in sublist]
        col = np.array(col)
        le = LabelEncoder()
        crt = le.fit_transform(col)
        temp = pd.Series(crt, name= namelist[ele])
        data = pd.concat([data, temp], axis=1)
        
        
# date_account_created
temp_fb = full.date_account_created
cate10 = np.vstack(temp_fb.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
names10 = ['year_ac','month_ac','day_ac']
# names10 = sorted(names10)
cate10 = pd.DataFrame(cate10, index=range(275547), columns=names10)

## date_first_booking
temp_dfb = full.date_first_booking
cate11 = np.vstack(temp_dfb.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
names11 = ['year_dfb','month_dfb','day_dfb']
# names11 = sorted(names11)
cate11 = pd.DataFrame(cate11, index=range(275547), columns=names11)
# print(cate11)

## timestamp_first_active
temp_tfa =full.timestamp_first_active
cate12 = np.vstack(temp_tfa.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
cate12 = cate12[:,0:3]
names12 = ['year_tfa','month_tfa','day_tfa']
cate12 = pd.DataFrame(cate12, index=range(275547), columns=names12)
# print(cate12)


data = pd.concat([data, cate10, cate11, cate12], axis=1)

full_1 = full.drop(['timestamp_first_active','gender','date_account_created','date_first_booking','signup_method','language','affiliate_channel',                    'affiliate_provider','first_affiliate_tracked','signup_app',                    'first_device_type','first_browser',],axis = 1)

full_data = pd.concat([data, full_1], axis=1)
print (full_data.shape)
print(full_data)


# In[79]:

# from sklearn import preprocessing
# converting_list = [3,5,7,8,9,10,11,12,13]
# namelist = list(balancefull)
# for ele in converting_list:
#     if ele == converting_list[0]:
#         col = balancefull.ix[:,ele:ele+1]
#         col = np.array(col)
#         col = [val for sublist in col for val in sublist]
#         col = np.array(col)
#         le = LabelEncoder()
#         crt = le.fit_transform(col)
#         data = pd.Series(crt, name= namelist[ele])
#     else:
#         col = balancefull.ix[:,ele:ele+1]
#         col = np.array(col)
#         col = [val for sublist in col for val in sublist]
#         col = np.array(col)
#         le = LabelEncoder()
#         crt = le.fit_transform(col)
#         temp = pd.Series(crt, name= namelist[ele])
#         data = pd.concat([data, temp], axis=1)
        
        
# # date_account_created
# temp_fb = balancefull.date_account_created
# cate10 = np.vstack(temp_fb.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
# names10 = ['year_ac','month_ac','day_ac']
# # names10 = sorted(names10)
# cate10 = pd.DataFrame(cate10, index=range(151004), columns=names10)

# ## date_first_booking
# temp_dfb = balancefull.date_first_booking
# cate11 = np.vstack(temp_dfb.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
# names11 = ['year_dfb','month_dfb','day_dfb']
# # names11 = sorted(names11)
# cate11 = pd.DataFrame(cate11, index=range(151004), columns=names11)
# # print(cate11)

# ## timestamp_first_active
# temp_tfa =balancefull.timestamp_first_active
# cate12 = np.vstack(temp_tfa.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
# cate12 = cate12[:,0:3]
# names12 = ['year_tfa','month_tfa','day_tfa']
# cate12 = pd.DataFrame(cate12, index=range(151004), columns=names12)
# # print(cate12)


# data = pd.concat([data, cate10, cate11, cate12], axis=1)

# balancefull_1 = balancefull.drop(['timestamp_first_active','gender','date_account_created','date_first_booking','signup_method','language','affiliate_channel',\
#                     'affiliate_provider','first_affiliate_tracked','signup_app',\
#                     'first_device_type','first_browser',],axis = 1)

# balancefull_data = pd.concat([data, balancefull_1], axis=1)
# print (balancefull_data.shape)
# print(balancefull_data)


# In[8]:

from sklearn.feature_selection import VarianceThreshold
def VarianceThreshold_selector(data):

    #Select Model
    selector = VarianceThreshold(1.4) #Defaults to 0.0, e.g. only remove features with the same value in all samples

    #Fit the Model
    selector.fit(data)
    features = selector.get_support(indices = True) #returns an array of integers corresponding to nonremoved features
    features = [column for column in data[features]] #Array of all nonremoved features' names

    #Format and Return
    selector = pd.DataFrame(selector.transform(data))
    selector.columns = features
    return selector

VarianceThreshold_selector(full_data)
# VarianceThreshold_selector(balancefull_data)


# # Train and Testing

# In[9]:

# Train and Testing 
training = full_data[0:213451]
testing = full_data[213451:]
print(training.shape)
print(testing.shape)
cv =  labels.reshape((len(labels), 1))


# In[69]:

# # Train and Testing 
# balance_training = balancefull_data[0:88908]
# testing = balancefull_data[88908:]
# print(balance_training.shape)
# print(testing.shape)
# balance_cv =  balancelabels.reshape((len(balancelabels), 1))


# In[10]:

# 'affiliate_provider' 'gender','signup_method', 'signup_app'
list(full_data)


# ### Feature Reduce

# In[20]:

redexp_full_data = full_data
redvar_full_data  = full_data
red_full_data = full_data


# In[21]:

# Feature Selection by data exploration
redexp_full_data = redexp_full_data.drop(['gender'],axis=1)
redexp_full_data = redexp_full_data.drop(['affiliate_provider'],axis=1)
redexp_full_data = redexp_full_data.drop(['month_ac'],axis=1)

print (redexp_full_data.shape)


# In[22]:

# Feature Selection by Variance Threshold¶
redvar_full_data = redvar_full_data.drop(['gender'],axis=1)
redvar_full_data = redvar_full_data.drop(['signup_app'],axis=1)
redvar_full_data = redvar_full_data.drop(['signup_method'],axis=1)
print (redvar_full_data.shape)


# In[23]:

# Feature Selection by data exploration and Variance Threshold
red_full_data = red_full_data.drop(['gender'],axis=1)
red_full_data = red_full_data.drop(['affiliate_provider'],axis=1)
red_full_data = red_full_data.drop(['month_ac'],axis=1)
red_full_data = red_full_data.drop(['signup_app'],axis=1)
red_full_data = red_full_data.drop(['signup_method'],axis=1)
print (red_full_data.shape)


# In[26]:

# Train and Testing 
cv =  labels.reshape((len(labels), 1))

## Full Data Set:
training = full_data[0:213451]
testing = full_data[213451:]

## redexp_full_data
redexp_training = redexp_full_data[0:213451]
redexp_testing = redexp_full_data[213451:]

## redvar_full_data
redvar_training = redvar_full_data[0:213451]
redvar_testing = redvar_full_data[213451:]

## red_full_data
red_training = red_full_data[0:213451]
red_testing = red_full_data[213451:]

print(training.shape)
print(testing.shape)
print("------------")
print(redexp_training.shape)
print(redexp_testing.shape)
print("------------")
print(redvar_training.shape)
print(redvar_testing.shape)
print("------------")
print(red_training.shape)
print(red_testing.shape)


# # Classification

# ## Decision Tree, Random Forest, XGboost Parameter Selection:

# ### Decision Tree: Choosing Optimal Parameters 

# In[37]:

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.cross_validation import KFold, cross_val_score
import numpy
import matplotlib.pyplot as plt

kf = KFold(len(cv), n_folds=5)
DTscore = []
DTvalidation = []
ParaDT = [1,5,10,25,50,75,100]
for i in ParaDT:
    DT = DecisionTreeClassifier(criterion="entropy", max_depth = i)
    DT.fit(training, cv)
    tempDT1 = DT.score(training,cv)
    DTscore.append(tempDT1)
    scores = cross_val_score(DT, training, cv, cv=kf)
    tempDT2= abs(scores.mean())
    DTvalidation.append(tempDT2)
    
plt.plot(ParaDT,DTscore)
plt.plot(ParaDT,DTvalidation)
plt.axis([0, 110, 0.5, 1.2])
plt.legend(['Training Score', 'Validation'], loc='lower left')
plt.show()


# ### Random Forest Choosing optimal parameters

# In[44]:

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
import numpy as np

kf = KFold(len(cv), n_folds=5)
RFscore = []
RFvalidation = []
ParaRF = [1,5,10,25,50,75,100]
for i in ParaRF:
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(training, cv.ravel())
    tempRF1 = clf.score(training,cv)
    RFscore.append(tempRF1)
    model = RandomForestClassifier()
    scores = cross_val_score(model, training, cv.ravel(), cv=kf)
    tempRF2= abs(scores.mean())
    RFvalidation.append(tempRF2)
    
plt.plot(ParaRF,RFscore)
plt.plot(ParaRF,RFvalidation)
plt.axis([0, 110, 0.5, 1.2])
plt.legend(['Training Score', 'Validation'], loc='lower left')
plt.show()


# ### Xgboost Choosing optimal parameters

# In[39]:

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
import numpy as np
from xgboost.sklearn import XGBClassifier

XGBscore = []
ParaXGB = [1,5,10,25,50]
for i in ParaXGB:
    xgb = XGBClassifier(max_depth=i, n_estimators=25,                     objective='multi:softprob', subsample=0.5, colsample_bytree=0.5)
    xgb.fit(training, cv.ravel())
    tempXGB1 = xgb.score(training,cv)
    XGBscore.append(tempXGB1)
    
plt.plot(ParaXGB,XGBscore)
plt.axis([0, 50, 0.5, 1.2])
plt.legend(['Training Score'], loc='lower left')
plt.show()


# ## Naive Bayes Modelling Function

# In[31]:

# Naive Bayes 
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

csvname = 0

def naiveBayse(training, cv, testing):
    global csvname
    NB = GaussianNB()
    NB.fit(training, cv.ravel())
    NBtrainscore = NB.score(training,cv)
   
    kf = KFold(len(cv), n_folds=5) # 5 folder cross validation
    scores = cross_val_score(NB, training,  cv.ravel(), cv=kf)
    NBvalidation = abs(scores.mean())
   
    NBy_pred = NB.predict_proba(testing)
   
    le = LabelEncoder()
    y = le.fit_transform(labels)
   
    idlist = [] #id list
    listcty = [] #countries list
   
    for i in range(len(testid)):
        idi = testid[i]
        idlist += [idi] * 5
        listcty += le.inverse_transform(np.argsort(NBy_pred[i])[::-1])[:5].tolist()

    NBsub = pd.DataFrame(np.column_stack((idlist, listcty)), columns=['id', 'country'])
    csvname = csvname + 1
    NBsub.to_csv('NBsub_%s.csv' % csvname,index=False)
    print("NBtrainscore", NBtrainscore)
    print "NBvalidation", NBvalidation

naiveBayse(redvar_training, cv, redvar_testing)



# ## Decision Tree Modelling Function

# In[35]:

# Decision Tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import preprocessing
from sklearn.cross_validation import KFold, cross_val_score
import numpy

def decisionTree(training, cv, testing):
    # Create linear regression object
    DT = DecisionTreeClassifier(criterion="entropy", max_depth = 15)
    # Train the model using the training sets
    DT.fit(training, cv)
    DTtrainscore = DT.score(training,cv) # training score

    kf = KFold(len(cv), n_folds=5) # 5 folder cross validation
    scores = cross_val_score(DT, training,  cv.ravel(), cv=kf)
    DTvalidation = abs(scores.mean())
    
    DTy_pred = DT.predict_proba(testing)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    idlist = [] #id list
    listcty = [] #countries list

    for i in range(len(testid)):
        idi = testid[i]
        idlist += [idi] * 5
        listcty += le.inverse_transform(np.argsort(DTy_pred[i])[::-1])[:5].tolist()

    DTsub = pd.DataFrame(np.column_stack((idlist, listcty)), columns=['id', 'country'])
    DTsub.to_csv('DTsub_%s.csv' % csvname,index=False)
    print("DTtrainscore", DTtrainscore)
    print("DTvalidation", DTvalidation)


decisionTree(red_training, cv, red_testing)


# ## Random Forest Modelling Function

# In[39]:

## Random Forest
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
import numpy as np


def randomforest(training, cv, testing):
    # Create linear regression object
    clf = RandomForestClassifier(n_estimators=8)
    # Train the model using the training sets
    clf.fit(training, cv.ravel())
    RFFtrainscore = clf.score(training,cv) # training score

    kf = KFold(len(cv), n_folds=5) # 5 folder cross validation
    scores = cross_val_score(clf, training,  cv.ravel(), cv=kf)
    RFvalidation = abs(scores.mean())
    
    RFy_pred = clf.predict_proba(testing)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    idlist = [] #id list
    listcty = [] #countries list

    for i in range(len(testid)):
        idi = testid[i]
        idlist += [idi] * 5
        listcty += le.inverse_transform(np.argsort(RFy_pred[i])[::-1])[:5].tolist()

    DTsub = pd.DataFrame(np.column_stack((idlist, listcty)), columns=['id', 'country'])
    DTsub.to_csv('RFsub_%s.csv' % csvname,index=False)
    print("RFFtrainscore", RFFtrainscore)
    print("RFvalidation", RFvalidation)

randomforest(red_training, cv, red_testing)


# ## xgboost

# In[43]:

import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score
import numpy as np
from xgboost.sklearn import XGBClassifier

def xxgboost(training, cv, testing):
    xgb = XGBClassifier(max_depth=6, n_estimators=25, objective='multi:softprob', subsample=0.5, colsample_bytree=0.5)
    
    xgb.fit(training, cv.ravel())
    XGBtrainscore = xgb.score(training,cv.ravel()) #Train Score

    kf = KFold(len(cv), n_folds=5) # 5 folder cross validation
    scores = cross_val_score(xgb, training,  cv.ravel(), cv=kf)
    XGBvalidation = abs(scores.mean())
    
    XGBy_pred = xgb.predict_proba(testing)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    idlist = [] #id list
    listcty = [] #countries list

    for i in range(len(testid)):
        idi = testid[i]
        idlist += [idi] * 5
        listcty += le.inverse_transform(np.argsort(XGBy_pred[i])[::-1])[:5].tolist()

    XGBsub = pd.DataFrame(np.column_stack((idlist, listcty)), columns=['id', 'country'])
    XGBsub.to_csv('XGsub_%s.csv' % csvname,index=False)
    print("XGBtrainscore", XGBtrainscore)
    print("XGBvalidation", XGBvalidation)
    
xxgboost(red_training, cv, red_testing)

