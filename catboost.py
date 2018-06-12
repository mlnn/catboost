
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier


# **Считываем данные**

# In[2]:


train_df = pd.read_csv('../../data/train.csv')


# In[3]:


train_df.head()


# In[4]:


test_df = pd.read_csv('../../data/test.csv')


# In[5]:


test_df.head()


# **Создаем всего один признак – маршрут**

# In[6]:


train_df['flight'] = train_df['Origin'] + '-->' + train_df['Dest']
test_df['flight'] = test_df['Origin'] + '-->' + test_df['Dest']


# **Запоминаем номера категориальных признаков**

# In[7]:


categ_feat_idx = np.where(train_df.drop('dep_delayed_15min', axis=1).dtypes == 'object')[0]
categ_feat_idx


# **Выделяем отложенную выборку**

# In[8]:


X_train = train_df.drop('dep_delayed_15min', axis=1).values
y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X_test = test_df.values


# In[9]:


X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 
                                                                test_size=0.3, 
                                                                random_state=17)


# In[10]:


ctb = CatBoostClassifier(random_seed=17)


# **Обучаем Catboost без настройки параметров, передав только индексы категориальных признаков.**

# In[11]:


get_ipython().run_cell_magic('time', '', 'ctb.fit(X_train_part, y_train_part,\n        cat_features=categ_feat_idx)')


# In[12]:


ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]


# **Получаем почти 0.75 ROC AUC на отложенной выборке.**

# In[13]:


roc_auc_score(y_valid, ctb_valid_pred)


# **Обучаем на всей выборке, делаем прогноз на тестовой, в соревновании получается результат 0.73008.**

# In[14]:


get_ipython().run_cell_magic('time', '', 'ctb.fit(X_train, y_train,\n        cat_features=categ_feat_idx)')


# In[15]:


ctb_test_pred = ctb.predict_proba(X_test)[:, 1]


# In[16]:


sample_sub = pd.read_csv('sample_submission.csv', index_col='id')
sample_sub['dep_delayed_15min'] = ctb_test_pred
sample_sub.to_csv('ctb_pred.csv')


# In[17]:


sample_sub.head()

