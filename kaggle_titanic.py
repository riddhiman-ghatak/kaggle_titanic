#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
train= pd.read_csv("C:/Users/riddh/Downloads/train (1).csv")
test= pd.read_csv("C:/Users/riddh/Downloads/test.csv")


# In[28]:


train.head()


# In[29]:


train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
full_data = [train, test]
for dataset in full_data:
    dataset['SibSp'] = dataset['SibSp'].fillna('1')
for dataset in full_data:
    dataset['Parch'] = dataset['Parch'].fillna('0')    


# In[30]:


full_data = [train, test]

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())    
    


# In[31]:


train.head()


# In[32]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
for dataset in full_data:
    dataset['Age'] = dataset['Age'].fillna(train['Age'].median())  
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].fillna('male')    


# In[33]:



   train.loc[ train['Fare'] <= 7.91, 'Fare'] 						        = 0
   train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
   train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
   train.loc[ train['Fare'] > 31, 'Fare'] 							        = 3



# In[34]:


train.head()


# In[35]:


train.loc[ train['Age'] <= 16, 'Age']= 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4 ;


# In[36]:


train.head()


# In[37]:


from sklearn.preprocessing import LabelEncoder
le_Sex = LabelEncoder()
le_Embarked = LabelEncoder()


# In[38]:


train['Sex_n'] = le_Sex.fit_transform(train['Sex'])
train['Embarked_n'] = le_Embarked.fit_transform(train['Embarked'])


# In[39]:


train.head()


# In[40]:


test.loc[ test['Fare'] <= 7.91, 'Fare']  = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[ test['Fare'] > 31, 'Fare'] =3
test.loc[ test['Age'] <= 16, 'Age']= 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age'] = 4 ;


# In[41]:


from sklearn.preprocessing import LabelEncoder
le_Sex = LabelEncoder()
le_Embarked = LabelEncoder()
test['Sex_n'] = le_Sex.fit_transform(test['Sex'])
test['Embarked_n'] = le_Embarked.fit_transform(test['Embarked'])


# In[42]:


test.head()


# In[43]:


drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Sex','Embarked']
train = train.drop(drop_elements, axis = 'columns')


# In[44]:


test  = test.drop(drop_elements, axis = 'columns')


# In[45]:


y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis='columns')
x_train = train.values 
x_test = test.values 


# In[46]:


import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC


# In[278]:


gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
xgb_predictions = gbm.predict(x_test)


# In[47]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=40)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)


# In[280]:


pred=pd.DataFrame(xgb_predictions)
df5= pd.read_csv("C:/Users/riddh/Downloads/test.csv")
datasets=pd.concat([df5['PassengerId'],pred],axis='columns')
datasets.columns=['PassengerId','Survived']
datasets.to_csv('submission_riddhiman_xgb.csv',index=False)


# In[48]:


pred2=pd.DataFrame(y_predicted)
df5= pd.read_csv("C:/Users/riddh/Downloads/test.csv")
datasets2=pd.concat([df5['PassengerId'],pred2],axis='columns')
datasets2.columns=['PassengerId','Survived']
datasets2.to_csv('submission_riddhiman_randomforest3.csv',index=False)


# In[ ]:




