
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Draw inline
get_ipython().magic(u'matplotlib inline')

# Set figure aesthetics
sns.set_style("white", {'ytick.major.size': 10.0})
sns.set_context("poster", font_scale=1.1)


# In[2]:

# Load the data into DataFrames
train_users = pd.read_csv('../input/train_users_2.csv')
test_users = pd.read_csv('../input/test_users.csv')


# In[4]:

print(train_users.shape[0],test_users.shape[0])


# In[6]:

# Merge train and test users
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

# Remove ID
users.drop('id',axis=1, inplace=True)

users.head(10)


# In[6]:

users.gender.replace('-unknown-', np.nan, inplace=True)


# In[15]:

users_nan = (users.isnull().sum() / users.shape[0]) * 100
users_nan[users_nan > 0].drop('country_destination')


# In[18]:

#check
print(int((train_users.date_first_booking.isnull().sum() / train_users.shape[0]) * 100))


# In[19]:

users.age.describe()


# In[20]:

print(sum(users.age > 100))
print(sum(users.age < 18))


# In[21]:

users[users.age > 100]['age'].describe()


# In[22]:

users[users.age < 18]['age'].describe()


# In[23]:

users.loc[users.age > 95, 'age'] = np.nan
users.loc[users.age < 13, 'age'] = np.nan


# In[24]:

categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'country_destination',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method'
]

for categorical_feature in categorical_features:
    users[categorical_feature] = users[categorical_feature].astype('category')


# In[25]:

users['date_account_created'] = pd.to_datetime(users['date_account_created'])
users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])
users['date_first_active'] = pd.to_datetime((users.timestamp_first_active // 1000000), format='%Y%m%d')


# In[26]:

series = pd.Series(users.gender.value_counts(dropna=False))


# In[28]:

series.plot.pie(figsize=(5, 5))


# In[37]:

women = sum(users['gender'] == 'FEMALE')
men = sum(users['gender'] == 'MALE')

female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100
male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100

# Bar width
width = 0.4

male_destinations.plot(kind='bar', width=width, color='#3CB371', position=0, label='Male', rot=0)
female_destinations.plot(kind='bar', width=width, color='#6495ED', position=1, label='Female', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage of the user')

sns.despine()
plt.show()


# In[42]:

destination_percentage = users.country_destination.value_counts() / users.shape[0] * 100
destination_percentage.plot(kind='bar',color='#20B2AA', rot=0)
# Using seaborn to plot
sns.countplot(x="country_destination", data=users, order=list(users.country_destination.value_counts().keys()))
plt.xlabel('Destination Country')
plt.ylabel('Percentage of the user')
# sns.despine()


# In[44]:

sns.kdeplot(users.age.dropna(), color='#20B2AA', shade=True)
plt.xlabel('Age')
plt.ylabel('Distribution of age')
sns.despine()


# In[45]:

age = 40

younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())
older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())

younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100
older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100

younger_destinations.plot(kind='bar', width=width, color='#3CB371', position=0, label='Youngers', rot=0)
older_destinations.plot(kind='bar', width=width, color='#6495ED', position=1, label='Olders', rot=0)

plt.legend()
plt.xlabel('Destination Country')
plt.ylabel('Percentage of the user')

sns.despine()
plt.show()


# In[50]:

df=users.date_account_created.value_counts()
plt.figure()
df.plot(colormap='winter')
plt.xlabel('First create account')


# In[51]:

df=users.date_first_active.value_counts()
plt.figure()
df.plot(colormap='winter')
plt.xlabel('Fisrt active account')

