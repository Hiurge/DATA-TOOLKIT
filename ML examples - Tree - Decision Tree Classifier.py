#!/usr/bin/env python
# coding: utf-8

# ## Decision Tree Classifier

# #### TL;DR

# In[5]:


from sklearn.tree import DecisionTreeClassifier # Algorithm


# In[6]:


ALG = DecisionTreeClassifier( 
                              class_weight=None, #
                              criterion='gini', #
                              max_depth=4, #
                              max_features=None, #
                              max_leaf_nodes=None, #
                              min_impurity_decrease=0.0, # 
                              min_impurity_split=None, #
                              min_samples_leaf=8, #
                              min_samples_split=2, #
                              min_weight_fraction_leaf=0.0, 
                              presort=False, #
                              random_state=0, #
                              splitter='best', #
                             )
ALG


# ### 1. Example with code

# In[7]:


import os # Files
import pandas as pd # Tables
import matplotlib.pyplot as plt # Plots
from sklearn.model_selection import train_test_split # ML

# Load cleaned and preprocessed CSV file as a dataframe.
fp = os.path.join('', 'tweets_sentiment.csv')    # File path
df = pd.read_csv(fp, sep='\t', encoding='utf-8') # Load as dataframe


# <b>Example problem</b>: Predict tweet sentiment basing on it's  nr of hashtags, retweet and like counts. 
# 
# <b>Example data</b>: consists of 3800 tweets obtained by twitter search API on phrases like psychology + AI (and simillar) saved and cleaned previously as a tweets_sentiment.csv file. Features:

# In[8]:


df.head(3)


# - <b>tweet</b>           - tweet text.
# - <b>hashtags</b>        - #hashtags in a tweet.
# - <b>hashtags_number</b> - number of hashtags.
# - <b>likes</b>           - number of tweet likes 
# - <b>retweets</b>        - number of times tweet have been shared.
# - <b>sentiment</b>       - score in range: -1.0 to 1.0 .
# - <b>sentiment_class</b> - score simplified to: Positive ( > 0) and Negative ( < 0).

# <b>Example code:</b>

# In[9]:


# Decision Tree Classifier

# Divide data into features(X) and labels(y).
y =  df.loc[ :, 'sentiment_class'] # column of labels to predict
X =  df.loc[ :, ['retweets', 'likes', 'hashtags_number']] # columns of features used to predict label

# Split both features(X) and labels(y) into training and testing datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Load and define Decision Tree Classifier.
DTC = DecisionTreeClassifier( 
                              class_weight=None, #
                              criterion='gini', #
                              max_depth=4, #
                              max_features=None, #
                              max_leaf_nodes=None, #
                              min_impurity_decrease=0.0, # 
                              min_impurity_split=None, #
                              min_samples_leaf=8, #
                              min_samples_split=2, #
                              min_weight_fraction_leaf=0.0, 
                              presort=False, #
                              random_state=0, #
                              splitter='best', #
                             )
# Fit data into model.
DTC.fit(X_train, y_train)

# Results                                                                                                 
accuracy_train = round(DTC.score(X_train, y_train), 2)
accuracy_test  = round(DTC.score(X_test,  y_test), 2)
predictions = DTC.predict(X_test) # an array.
probabilities = DTC.predict_proba(X_test) # an array.

# Display results.
print('Accuracy - train: {}\nAccuracy - test:  {}\nFirst three predictions (of {}): {} ...\nFirst three propabilities (of {}): {} ...'.format(accuracy_train, accuracy_test, len(predictions), predictions[:3], len(probabilities), probabilities[:3]))
plt.scatter(y_test, predictions)
plt.show()


# ### 2. Key info
# 
# - [ADD MORE],
# - (To be updated.)

# ### 3. Template

# In[ ]:


import os # Get file
import pandas as pd # Read as pandas table; dataframe (df).
from sklearn.model_selection import train_test_split # Train/Test set divide.
from sklearn.tree import DecisionTreeClassifier # Algorithm
import matplotlib.pyplot as plt # Plots

# You fill three lines below.
# ---------------------------
file_name   = 'your file_name.csv' # csv file in same dir  as this notebook.
predit_what = 'column_name' # The label to predict.
based_on    = ['column_name', 'column_name'] # The features to use in this quest.

# You may wany to change full file path / use existing dataframe. 
fp = os.path.join('', file_name) # fp = 'home/data/file_path.csv'
df = pd.read_csv(fp, sep='\t', encoding='utf-8') # df = my_df 


# Decision Tree Classifier

# Divide data into features(X) and labels(y).
X =  df.loc[ :, based_on]    # features
y =  df.loc[ :, predit_what] # label

# Split both features(X) and labels(y) into training and testing datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Load and define Decision Tree Classifier model.
DTC = DecisionTreeClassifier( 
                              class_weight=None, #
                              criterion='gini', #
                              max_depth=4, #
                              max_features=None, #
                              max_leaf_nodes=None, #
                              min_impurity_decrease=0.0, # 
                              min_impurity_split=None, #
                              min_samples_leaf=8, #
                              min_samples_split=2, #
                              min_weight_fraction_leaf=0.0, 
                              presort=False, #
                              random_state=0, #
                              splitter='best', #
                             )

# Fit data into model.
DTC.fit(X_train, y_train, sample_weight=None)

# Results.
r_squared_train = round(DTC.score(X_train, y_train), 2)
r_squared_test  = round(DTC.score(X_test,  y_test), 2)
predictions = DTC.predict(X_test) # an array.
probabilities = DTC.predict_proba(X_test) # an array.

# Display results.
print('Accuracy - train: {}\nAccuracy - test:  {}\nFirst three predictions (of {}): {} ...\nFirst three propabilities (of {}): {} ...'.format(accuracy_train, accuracy_test, len(predictions), predictions[:3], len(probabilities), probabilities[:3]))
plt.scatter(y_test, predictions)
plt.show()


# #### Concise

# In[ ]:


import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

fp = os.path.join('', 'your file_name.csv')
df = pd.read_csv(fp, sep='\t', encoding='utf-8')

X =  df.loc[ :, ['feature_column_name', 'feature_column_name']]
y =  df.loc[ :, 'label_column_name']
X_train, X_test, y_train, y_test = train_test_split(X, y)

DTC = DecisionTreeClassifier(max_depth=4, min_samples_leaf= 8).fit(X_train, y_train)

r_squared_train = round(DTC.score(X_train, y_train), 2)
r_squared_test  = round(DTC.score(X_test,  y_test), 2)
predictions = DTC.predict(X_test)
probabilities = DTC.predict_proba(X_test)

print('Accuracy - train: {}\nAccuracy - test:  {}\nFirst three predictions (of {}): {} ...\nFirst three propabilities (of {}): {} ...'.format(accuracy_train, accuracy_test, len(predictions), predictions[:3], len(probabilities), probabilities[:3]))
plt.scatter(y_test, predictions)
plt.show()


# ### 4. More
# 
# To be updated.

#  

# By Luke, 13 II 2019.
