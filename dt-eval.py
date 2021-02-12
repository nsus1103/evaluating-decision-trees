#!/usr/bin/env python
# coding: utf-8

# In[183]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from random import randint
from sklearn.metrics import confusion_matrix as cm
import seaborn as sns


# In[254]:


raw2 = pd.read_csv('Biomechanical_Data_2classes.csv')
raw3 = pd.read_csv('Biomechanical_Data_3classes.csv')


# In[3]:


raw2['class'].value_counts()


# In[4]:


raw3['class'].value_counts()


# In[5]:


raw2.dtypes


# In[6]:


raw2['class'] = raw2['class'].map({'Normal': 1, 'Abnormal':0})


# In[238]:


raw2.head()


# In[8]:


raw_merged = pd.merge(left = raw2, right = raw3,left_on=['pelvic_incidence', 'lumbar_lordosis_angle',
        'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis'], right_on= ['pelvic_incidence', 'lumbar_lordosis_angle',
        'sacral_slope', 'pelvic_radius', 'degree_spondylolisthesis'], how = 'inner')


# In[9]:


raw_merged.columns, raw2.columns, raw3.columns


# In[10]:


raw_merged[['class_x', 'class_y']].value_counts()


# In[211]:


raw2[raw2['degree_spondylolisthesis']<0].shape


# In[212]:


df2 = raw2.copy()


# In[213]:


X2 = df2.drop('class', axis=1).copy()
y2 = df2['class'].copy()


# In[214]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=70, random_state=42 )


# In[215]:


y2_train.value_counts()


# In[216]:


X2_train.shape, X2_test.shape, y2_train.shape, y2_test.shape


# In[217]:


k = [8, 16, 32]
dt1 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=8)
dt2 = DecisionTreeClassifier(criterion = 'entropy',min_samples_leaf=16)
dt3 = DecisionTreeClassifier(criterion = 'entropy',min_samples_leaf=32)

model1 = dt1.fit(X2_train, y2_train)
model2 = dt2.fit(X2_train, y2_train)
model3 = dt3.fit(X2_train, y2_train)


# ## Problem 1

# In[218]:


fig = plt.figure(figsize=(25,20))
tree1 = tree.plot_tree(dt1, 
                       feature_names=X2.columns, 
                       class_names=['Abnormal','Normal'], 
                       node_ids=True, 
                       filled=True)


# In[219]:


fig = plt.figure(figsize=(25,20))
tree2 = tree.plot_tree(dt2, 
                       feature_names=X2.columns, 
                       class_names=['Abnormal','Normal'], 
                       node_ids=True, 
                       filled=True)


# In[220]:


fig = plt.figure(figsize=(25,20))
tree3 = tree.plot_tree(dt3, 
                       feature_names=X2.columns, 
                       class_names=['Abnormal','Normal'], 
                       node_ids=True, 
                       filled=True)


# In[222]:


class color:
   BOLD = '\033[1m'
   END = '\033[0m'
    
# TO export the decision tree summary as text
from sklearn.tree import export_text
tree1_rules = export_text(dt1, feature_names=list(X2.columns))
tree2_rules = export_text(dt2, feature_names=list(X2.columns))
tree3_rules = export_text(dt3, feature_names=list(X2.columns))

print(color.BOLD + 'Decision Tree with min_samples_leaf =   8' + color.END, '\n\n', tree1_rules,'\n',
      color.BOLD + 'Decision Tree with min_samples_leaf =  16' + color.END, '\n\n', tree2_rules,'\n',
      color.BOLD + 'Decision Tree with min_samples_leaf =  32' + color.END, '\n\n', tree3_rules)


# ## Problem 1a.
# 
# ### Comparative Analysis of the Decision Trees
# 
# Following is a summary and the analysis of the three decision trees obtained by changing the values of the hyperparameter _'min_samples_leaf'_ to {_8, 16, 32_} [min_samples_leaf is a hyperparameter that gievrns the minimum number of datapoints to be present in a node after a split is carried out. Its function is make the model robust to outliers and, thus, reduce overfitting ]. Thus, it is reasonable to expect a Decision Tree with lower min_samples_leaf to have more nodes and a larger depth.<br> The criteria used to calculate the information gain is the **'entropy'**
# 
# To summarize the decision trees, we examine first the decision tree itself by looking at the number of nodes created(splits), the values at which the split is carried out, the depth of the trees and the information gain at eaxh stage.
# 
# 
# | Min Points per leaf node|8|16|32|
# |-------------------------|-|--|--|
# | Total Number of nodes   |26|14|8|
# |Maximum Depth            |7|4 |3 |
# 
# <br>
# 
# It is worth noting that all the three trees made the first split at _degree_spondylolisthesis <= 20.09_ . <br>All the nodes to the right of the split i.e. with _degree_spondylolisthesis >= 20.09_ are classified as **Abnormal** in all the trees. This suggest heavy correlation between high values of _degree_spondylolisthesis_ and the diagnosis. <br>
# Looking at ths information, the trees with greater depth and higher number of nodes do not provide any additinal information through further splits. However, the difference in the trees arises for values of _degree_spondylolisthesis <= 20.09_.<br>The second split for all the three trees is common too at _pelvic_tilt numeric <= 10.72_. For the Third decision tree, the points with _pelvic_tilt numeric >= 10.72 area classified as **Abnormal** while the others area classified as **Normal**. In the trees 1 and 2, however, there is a further split at _lumbar_lordosis_angle_ to futher split the classes. **This is a direct consequene of allowing the tree to split further because of the lower restriction on the number of points in the leaf node. While the 3rd tree with a _min_samples_leaf=32_ has no such opportunity.**
# 
# To assess which of the trees is the best, we must look towards the information gained by the trees by being able to perform more splits compared to the Tree 3. Tree 1 and 2 do appear to be finding more information when compared to the Tree 3 (by observing node #3 in both the trees), however, Tree 1 does not gain any forther information by making further splits as the lower nodes are still classified as **Normal**. Therefore, It would appear that the **Tree 2 is the best among the three on the basis of information gained in minimum tree size**. 
#  

# In[225]:


y2_1_pred = model1.predict(X2_test)
y2_2_pred = model2.predict(X2_test)
y2_3_pred = model3.predict(X2_test)


# In[237]:


X2_train.shape


# In[239]:



display('CONFUSION MATRIX FOR THE THREE DECISION TREES','Decision Tree with min_samples_leaf =   8', 
      pd.DataFrame(cm(y2_test, y2_1_pred), 
                   index=['Actual Abnormal', 'Actual Normal'], 
                   columns=['Predicted Abnormal', 'Predicted Normal']),
      'Decision Tree with min_samples_leaf =   16',
      pd.DataFrame(cm(y2_test, y2_2_pred), 
                   index=['Actual Abnormal', 'Actual Normal'], 
                   columns=['Predicted Abnormal', 'Predicted Normal']),
      'Decision Tree with min_samples_leaf =   32' ,
      pd.DataFrame(cm(y2_test, y2_3_pred), 
                   index=['Actual Abnormal', 'Actual Normal'], 
                   columns=['Predicted Abnormal', 'Predicted Normal']))


# In[240]:


classes = ['Abnormal', 'Normal']
print(classification_report(y2_test, y2_1_pred, target_names=classes), classification_report(y2_test, y2_2_pred, target_names=classes), classification_report(y2_test, y2_3_pred, target_names=classes))


# ## Problem 1b.
# 
# #### __Confusion Matrix__
# 
# 
# ##### **Decision Tree with min_samples_leaf =   8 (Decision Tree 1)**
# 
# |                |    Predicted Abnormal |	Predicted Normal |
# |----------------|-----------------------|-------------------|
# |Actual Abnormal |	            44 	     | 6                 | 
# | Actual Normal  |                    9  |	11               | 
# 
# ##### **Decision Tree with min_samples_leaf =   16 (Decision Tree 2)**
# 
# |                |    Predicted Abnormal |	Predicted Normal |
# |----------------|-----------------------|-------------------|
# |Actual Abnormal |	            48 	     | 2                 | 
# | Actual Normal  |                    14 |	6                | 
# 
# 
# ##### **Decision Tree with min_samples_leaf =   32 (Decision Tree 3)**
# 
# |                |    Predicted Abnormal |	Predicted Normal |
# |----------------|-----------------------|-------------------|
# |Actual Abnormal |	            47 	     | 3                 | 
# | Actual Normal  |                    6  |	14               | 
#            
#            
# 
# #### __Model Scores__
# ##### **Decision Tree with min_samples_leaf =   8 (Decision Tree 1)**
# 
# 
# |         | precision  |  recall | f1-score |  support |
# |---------|------------|---------|----------|----------|
# |Abnormal |      0.83  |    0.88 |    0.85  |     50   |
# |  Normal |      0.65  |    0.55 |    0.59  |     20   |
# |Accuracy |            |         |     0.79 |       70 |
# 
# ##### **Decision Tree with min_samples_leaf =   16 (Decision Tree 2)**
# 
# |         | precision  |  recall | f1-score |  support |
# |---------|------------|---------|----------|----------|
# |Abnormal |      0.77  |    0.96 |    0.86  |     50   |
# |  Normal |      0.75  |    0.40 |    0.43  |     20   |
# |Accuracy |            |         |     0.77 |     70   |
# 
# 
# 
# ##### **Decision Tree with min_samples_leaf =   32 (Decision Tree 3)**
# 
# |         | precision  |  recall | f1-score |  support |
# |---------|------------|---------|----------|----------|
# |Abnormal |      0.89  |    0.94 |    0.91  |     50   |
# |  Normal |      0.82  |    0.70 |    0.76  |     20   |
# |Accuracy |            |         |    0.87  |     70   |
# 
# 
# #####  Analysis:
# Observing the above data, we can see that the decision tree 3 (min_samples_leaf=32) has the highest accuracy and f1-score among all the three models and thus can be easily concluded to be the best model.<br><br> However, looking closer at the confusion matrix, to guage the actual potential of the model in correctly identifying the abnormal cases, we come across an entirely different story, we can see that the decision tree 1 has misclassified 6 Abnormal cases as Normal, while the misclassification error of the decision tree 3 (with min_samples_leaf=32) is 3. <br><br>Considering the cost of misclassification in this case i.e. diagnosing a medical condition, misclassification carries with it huge costs for both the patient, the health care institutions like the Hospitals and Insurance agencies. <br><br>Therefore, concluding that the Decision Tree 1 is the best can have grave consequences in real life. By this criterion, one may conclude however that the Decision Tree 2 (#misclassified abnormal points = 2) is the best classifier model among the three as it has the **lowest rate of False Negatives**.

# In[ ]:


probas = dt1.predict_proba(X2_train)


# In[79]:


import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_cumulative_gain(y2_train, probas)
plt.show()


# ## Problem 2

# In[256]:


raw3['class'].value_counts()


# In[257]:


raw3


# In[258]:


raw3['class'] = raw3['class'].map({'Spondylolisthesis':0, 'Normal': 1, 'Hernia': 2})

raw3.head()

df3 = raw3.copy()

X3 = df3.drop('class', axis=1).copy()
y3 = df3['class'].copy()


# In[259]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=70, random_state=7 )

y3_train.value_counts()

X3_train.shape, X3_test.shape, y3_train.shape, y3_test.shape


# In[260]:


dt1 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=8)
dt2 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=16)
dt3 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=32)

model1 = dt1.fit(X3_train, y3_train)
model2 = dt2.fit(X3_train, y3_train)
model3 = dt3.fit(X3_train, y3_train)


# In[261]:


fig = plt.figure(figsize=(25,20))
tree1 = tree.plot_tree(dt1, 
                       feature_names=X3.columns, class_names=['Spondylolisthesis', 'Normal', 'Hernia'], 
                       node_ids=True, 
                       filled=True)


# In[262]:


fig = plt.figure(figsize=(25,20))
tree2 = tree.plot_tree(dt2, 
                       feature_names=X3.columns, 
                       class_names=['Spondylolisthesis', 'Normal', 'Hernia'],
                       node_ids=True,
                       filled=True)


# In[263]:


fig = plt.figure(figsize=(25,20))
tree3 = tree.plot_tree(dt3, 
                       feature_names=X3.columns, 
                       class_names=['Spondylolisthesis', 'Normal', 'Hernia'], 
                       node_ids=True,
                       filled=True)


# ## Problem 2a.
# 
# ### Comparative Analysis of the Decision Trees
# 
# Following is a summary and the analysis of the three decision trees obtained by changing the values of the hyperparameter _'min_samples_leaf'_ to {_8, 16, 32_} [min_samples_leaf is a hyperparameter that gievrns the minimum number of datapoints to be present in a node after a split is carried out. Its function is make the model robust to outliers and, thus, reduce overfitting ]. The criteria used to calculate the information gain is the **'entropy'**
# 
# To summarize the decision trees, we examine first the decision tree itself by looking at the number of nodes created(splits), the values at which the split is carried out, the depth of the trees and the information gain at eaxh stage.
# 
# 
# | Min Points per leaf node|8|16|32|
# |-------------------------|-|--|--|
# | Total Number of nodes   |28|16|10|
# |Maximum Depth            |7|5 |4 |
# 
# <br>
# 
# It is worth noting that all the three trees made the first split at _degree_spondylolisthesis <= 15.15_ . <br>All the nodes to the right of the split i.e. with _degree_spondylolisthesis >= 15.15_ are classified as **Spondylolisthesis** in all the trees. This suggest heavy correlation between high values of _degree_spondylolisthesis_ and the diagnosis. <br>
# Looking at ths information, the trees with greater depth and higher number of nodes do not provide any additinal information through further splits. However, the difference in the trees arises for values of _degree_spondylolisthesis <= 16.08_.<br>The second split for all the three trees is common too at _sacral _slope_.
# 
# The entropy at the leaf nodes is high in the Third DT because of high impurity at the leaf nodes. It could be explained in the Decision Tree 3 as a case of underfitting however, for Decision Tree 1 could be overfitting which can explain the low entropy of the nodes. <br>It is also interesting to note that in the Decision Tree 1 & 2, a split decision at node 8 (in Decision Tree 2) and node 15 ( in Decision tree 1) causes the model to classify certain points as **Normal**. It may be that the model is overfitting since the resulting nodes have high impurity or it may be a discernible opportunity to improve the model which the Decision Tree 3 did not get because of the constraint. 

# In[241]:


y3_1_pred = model1.predict(X3_test)
y3_2_pred = model2.predict(X3_test)
y3_3_pred = model3.predict(X3_test)


# In[242]:



display('CONFUSION MATRIX FOR THE THREE DECISION TREES','Decision Tree with min_samples_leaf =   8', 
    pd.DataFrame(cm(y3_test, y3_1_pred), 
                 index=['Actual Spondylolisthesis', 'Actual Normal', 'Actual Hernia'], 
                 columns=['Predicted Spondylolisthesis', 'Predicted Normal', ' Predicted Hernia']),
    'Decision Tree with min_samples_leaf =   16',
    pd.DataFrame(cm(y3_test, y3_2_pred), 
                 index=['Actual Spondylolisthesis', 'Actual Normal', 'Actual Hernia'], 
                 columns=['Predicted Spondylolisthesis', 'Predicted Normal', ' Predicted Hernia']),
    'Decision Tree with min_samples_leaf =   32' ,
    pd.DataFrame(cm(y3_test, y3_3_pred), 
                 index=['Actual Spondylolisthesis', 'Actual Normal', 'Actual Hernia'], 
                 columns=['Predicted Spondylolisthesis', 'Predicted Normal', ' Predicted Hernia']))


# In[243]:


classes = ['Spondylolisthesis', 'Normal', 'Hernia']
print(classification_report(y3_test, y3_1_pred, target_names=classes), 
      classification_report(y3_test, y3_2_pred, target_names=classes), 
      classification_report(y3_test, y3_3_pred, target_names=classes))


# ## Problem 2b.
# 
# #### __Confusion Matrix__
# 
# 
# ##### **Decision Tree with min_samples_leaf =   8 (Decision Tree 4)**
# 
# |                         |Predicted Spondylolisthesis |Predicted Normal|Predicted Hernia |
# |-------------------------|----------------------------|----------------|-----------------|
# |Actual Spondylolisthesis |	            35 	           | 1              | 0               | 
# | Actual Normal           |                    2       |	19          | 4               |
# | Actual Hernia           |                    1       |	6           | 2               |
# 
# ##### **Decision Tree with min_samples_leaf =   16 (Decision Tree 5)**
# 
# |                         |Predicted Spondylolisthesis |Predicted Normal|Predicted Hernia |
# |-------------------------|----------------------------|----------------|-----------------|
# |Actual Spondylolisthesis |	            35 	           | 1              | 0               | 
# | Actual Normal           |                    2       |	15          | 8               |
# | Actual Hernia           |                    1       |	3           | 5               |
# 
# 
# ##### **Decision Tree with min_samples_leaf =   32 (Decision Tree 6)**
# 
# |                         |Predicted Spondylolisthesis |Predicted Normal|Predicted Hernia |
# |-------------------------|----------------------------|----------------|-----------------|
# |Actual Spondylolisthesis |	            35 	           | 1              | 0               | 
# | Actual Normal           |                    2       |	20          | 3               |
# | Actual Hernia           |                    1       |	5           | 3               |
#            
# 
# #### __Model Scores__
# ##### **Decision Tree with min_samples_leaf =   8 (Decision Tree 4)**
# 
# 
# |                  | precision  |  recall | f1-score |  support |
# |------------------|------------|---------|----------|----------|
# |Spondylolisthesis |      0.92  |    0.97 |    0.95  |     36   |
# |  Normal          |      0.73  |    0.76 |    0.75  |     25   |
# |  Hernia          |      0.33  |    0.22 |    0.27  |     9    |
# |Accuracy          |            |         |    0.80  |     70   |
# 
# ##### **Decision Tree with min_samples_leaf =   16 (Decision Tree 5)**
# 
# |                  | precision  |  recall | f1-score |  support |
# |------------------|------------|---------|----------|----------|
# |Spondylolisthesis |      0.92  |    0.97 |    0.95  |     36   |
# |  Normal          |      0.79  |    0.60 |    0.68  |     25   |
# |  Hernia          |      0.33  |    0.56 |    0.45  |     9    |
# |Accuracy          |            |         |    0.79  |     70   |
# 
# 
# ##### **Decision Tree with min_samples_leaf =   32 (Decision Tree 6)**
# 
# |                  | precision  |  recall | f1-score |  support |
# |------------------|------------|---------|----------|----------|
# |Spondylolisthesis |      0.92  |    0.97 |    0.95  |     36   |
# |  Normal          |      0.77  |    0.80 |    0.78  |     25   |
# |  Hernia          |      0.50  |    0.33 |    0.40  |     9    |
# |Accuracy          |            |         |    0.83  |     70   |
# 
# 
# 
# #####  Analysis:
# 
# ||Decision Tree 1|Decision Tree 2|Decision Tree 3|
# |-|-|-|-|
# |Number of points misclassified as Normal|7|4|6|
# |Number of points misclassified the wrong disease|1|1|1|
# 
# Observing the above data, we can see that the decision tree 1 (min_samples_leaf=32) has the highest accuracy and f1-score among all the three models and thus can be easily concluded to be the best model.<br><br> However, looking closer at the confusion matrix, to guage the actual potential of the model in correctly identifying the abnormal cases, we come across an entirely different story, we can see that the decision tree 1 has misclassified 7 Abnormal cases as Normal, while the misclassification error (Classifying Abnormal as Normal) of the decision tree 3 (with min_samples_leaf=32) is 6. <br>Considering the cost of misclassification in this case i.e. diagnosing a medical condition, misclassification carries with it huge costs for both the patient, the health care institutions like the Hospitals and Insurance agencies. <br>Therefore, concluding that the Decision Tree 1 or 3 is the best can have grave consequences in real life. By this criterion, one may conclude however that the Decision Tree 2 is the best classifier model among the three as it has the lowest rate of False Negatives.

# ## Problem 3

# ### Dropping the feature 'degree_spondylolisthesis'

# In[162]:


df2 = raw2.drop('degree_spondylolisthesis', axis=1)


# In[201]:


df2.head()


# In[164]:


X2 = df2.drop('class', axis=1).copy()
y2 = df2['class'].copy()

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=70, random_state=42 )

y2_train.value_counts()

X2_train.shape, X2_test.shape, y2_train.shape, y2_test.shape

k = [8, 16, 32]
dt1 = DecisionTreeClassifier(criterion = 'entropy', min_samples_leaf=8)
dt2 = DecisionTreeClassifier(criterion = 'entropy',min_samples_leaf=16)
dt3 = DecisionTreeClassifier(criterion = 'entropy',min_samples_leaf=32)

model1 = dt1.fit(X2_train, y2_train)
model2 = dt2.fit(X2_train, y2_train)
model3 = dt3.fit(X2_train, y2_train)


# ### Problem 3a.
# 
# Having dropped the column 'degree_spondylolisthesis', we train the decision tree classidier on the new dataset and Export the splitting criteria at each node.

# In[165]:


class color:
   BOLD = '\033[1m'
   END = '\033[0m'
    
# TO export the decision tree summary as text
from sklearn.tree import export_text
tree1_rules = export_text(dt1, feature_names=list(X2.columns))
tree2_rules = export_text(dt2, feature_names=list(X2.columns))
tree3_rules = export_text(dt3, feature_names=list(X2.columns))

print(color.BOLD + 'Decision Tree with min_samples_leaf =   8' + color.END, '\n\n', tree1_rules,'\n',
      color.BOLD + 'Decision Tree with min_samples_leaf =  16' + color.END, '\n\n', tree2_rules,'\n',
      color.BOLD + 'Decision Tree with min_samples_leaf =  32' + color.END, '\n\n', tree3_rules)


# In[174]:


fig = plt.figure(figsize=(25,20))
tree3 = tree.plot_tree(dt1, 
                       feature_names=X2.columns, 
                       class_names=['Abnormal','Normal'], 
                       node_ids=True, 
                       filled=True)


# In[167]:


fig = plt.figure(figsize=(25,20))
tree3 = tree.plot_tree(dt2, 
                       feature_names=X2.columns, 
                       class_names=['Abnormal','Normal'], 
                       node_ids=True, 
                       filled=True)


# In[168]:


fig = plt.figure(figsize=(25,20))
tree3 = tree.plot_tree(dt3, 
                       feature_names=X2.columns, 
                       class_names=['Abnormal','Normal'], 
                       node_ids=True, 
                       filled=True)


# ## Problem 3a.
# 
# ### Comparative Analysis of the Decision Trees
# 
# Following is a summary and the analysis of the three decision trees obtained by changing the values of the hyperparameter _'min_samples_leaf'_ to {_8, 16, 32_} [min_samples_leaf is a hyperparameter that gievrns the minimum number of datapoints to be present in a node after a split is carried out. Its function is make the model robust to outliers and, thus, reduce overfitting ]. Thus, it is reasonable to expect a Decision Tree with lower min_samples_leaf to have more nodes and a larger depth.<br>The criteria used to calculate the information gain is the **'entropy'**
# 
# To summarize the decision trees, we examine first the decision tree itself by looking at the number of nodes created(splits), the values at which the split is carried out, the depth of the trees and the information gain at eaxh stage.
# 
# 
# | Min Points per leaf node|8|16|32|
# |-------------------------|-|--|--|
# | Total Number of nodes   |36|18|10|
# |Maximum Depth            |7|6 |4 |
# 
# <br>
# 
# It is worth noting that all the three trees made the first split at _pelvic_incidence > 69.48_ . <br>All the nodes to the right of the split i.e. with _pelvic_incidence > 69.48_ are classified as **Abnormal** in all the trees. This suggest heavy correlation between high values of _pelvic_incidence_ and the diagnosis. <br>
# 
# Looking at ths information, the trees with greater depth and higher number of nodes do not provide any additinal information through further splits. For example, it is important to observe that in the Decision Tree 1, for the points _pelvic_incidence > 69.48_, although the decision tree splits multiple times (from the node 30), the classification for all the points is **Abnormal**, making those splits redundant. The same can be observed at other nodes as well like node 2 and 18. This may lead to overfitting. <br>
# 
# The same issue can be seen in the Decision Tree 2, even though it is far less pronounced compared to the previos Decision Tree. The split at nodes 2, 8, 13, 6 do not add any new information to the model and are thus redundant. 
# 
# The third Decision Tree with a significantly higher restriction on the minimum number of points on the leaf node, does not grow very large and stops at max dept=4. The splits are more or less the same as earlier tree suggesting that the tree has captured most if not all the information in the dataset.
# 
# 
# To assess which of the trees is the best, we must look towards the information gained by the trees by being able to perform more splits compared to the Tree 3. Tree 1 and 2 do appear to be finding more information when compared to the Tree 3 (by observing node #3 in both the trees), however, Tree 1 does not gain any forther information by making further splits as the lower nodes are still classified as **Normal**. Therefore, It would appear that the **Tree 3 is the best among the three on the basis of information gained in minimum tree size**. We can make further evaluation once we observe the test statistics.

# In[169]:


y2_1_pred = model1.predict(X2_test)
y2_2_pred = model2.predict(X2_test)
y2_3_pred = model3.predict(X2_test)


# In[268]:


class color:
   BOLD = '\033[1m'
   END = '\033[0m'
    
print(color.BOLD + 'CONFUSION MATRIX FOR THE THREE DECISION TREES' + color.END)

display('Decision Tree with min_samples_leaf =   8', 
        pd.DataFrame(cm(y2_test, y2_1_pred), 
                     index=['Actual Abnormal', 'Actual Normal'], 
                     columns=['Predicted Abnormal', 'Predicted Normal']),
        'Decision Tree with min_samples_leaf =   16',
        pd.DataFrame(cm(y2_test, y2_2_pred), 
                     index=['Actual Abnormal', 'Actual Normal'], 
                     columns=['Predicted Abnormal', 'Predicted Normal']),
        'Decision Tree with min_samples_leaf =   32' ,
        pd.DataFrame(cm(y2_test, y2_3_pred), 
                     index=['Actual Abnormal', 'Actual Normal'], 
                     columns=['Predicted Abnormal', 'Predicted Normal']))


# In[171]:


classes = ['Abnormal', 'Normal']
print(classification_report(y2_test, y2_1_pred, target_names=classes), 
      classification_report(y2_test, y2_2_pred, target_names=classes), 
      classification_report(y2_test, y2_3_pred, target_names=classes))


# Observing the metrics obtained on the test dataset, one can easily notice that the accuracy of the three models is lower than the earlier models trined on the complete dataset, sugesting that a lot of predictive potential was lost on dropping the column _degree_spondylolisthesis_. <br>
# 
# The decision tree 2 outperforms noth the others trees sugesting that the decision tree 1 (with min_samples_leaf=8) must be overfitting the data while decision tree 3 (with min_samples_leaf=32) might be underfitting the data. The number of misclassifications can be observed from the confusion matrices and we can conclude that the deicsion tree 2 outperforms the other models (FalsePositive(FP)=11, FalseNegative(FN)=6) vs (FP=11, FN=9) for the Decision Tree 1 and (FP=11, FN=10) for the decision tree 3. 
# 
# <br>THus it is reasonable to conclude that the decision tree 2 is the best among three even with the truncated dataset.

# ## Problem 3b.
# 
# We observe that revised splitting decision at the root node has been carried at the 'pelvic_incidence' (><69.48). This decision criteria was not present at all in any tree in the previous iterations with the complete dataset. This is interesting and needs to be investigated further. This might suggest a high correlation between the features 'degree_spondylolisthesis' and 'pelvic_incidence' thus by splitting the dataset at 'degree_spondylolisthesis', the earlier trees might have gleaned all the information that could have been recieved from the 'pelvic_incidence' feature. We will investigate this through a correlation matrix.

# In[200]:


corr = raw2.drop('class', axis=1).corr(method = 'pearson')
sns.heatmap(corr, cmap = sns.diverging_palette(230, 20, as_cmap=True))


# In[198]:


raw2.drop('class', axis=1).corr(method = 'pearson')


# We see that the feature 'pelvic_incidence' is highly correlated with the feature 'degree_spondylolisthesis', in fact, 'pelvic_incidence' has the highest correlation coefficients among all the features to the 'degree_spondylolisthesis', which, in retrospect, seems logical. The Decision tree would perform the split at the next best feature, which in this case happens to be 'pelvic_incidence'.
# 
# The values of accurancy and precision are also much worse that in the previous cases which can again be attributed to information lost by truncating the dataset, which makes sense because removing the feature with the highest predictive potential would imapct the results negatively. 
# 
# 
# The misclassification error is significantly higher in these models as well.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[88]:


# Code to summarize the Decision Tree 
#(Source: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html)

n_nodes = dt1.tree_.node_count
children_left = dt1.tree_.children_left
children_right = dt1.tree_.children_right
feature = dt1.tree_.feature
threshold = dt1.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has {n} nodes and has "
      "the following tree structure:\n".format(n=n_nodes))
for i in range(n_nodes):
    if is_leaves[i]:
        print("{space}node={node} is a leaf node.".format(
            space=node_depth[i] * "\t", node=i))
    else:
        print("{space}node={node} is a split node: "
              "go to node {left} if {feature} <= {threshold}" 
              "else to node {right}.".format(
                  space=node_depth[i] * "\t",
                  node=i,
                  left=children_left[i],
                  feature=X3.columns[feature[i]],
                  threshold=threshold[i],
                  right=children_right[i]))


# In[25]:


# To export the decision tree nodes as IF...ELSE
from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


# In[26]:


tree_to_code(dt1, X2.columns)


# In[33]:


# TO export the decision tree summary as text
from sklearn.tree import export_text
tree1_rules = export_text(dt1, feature_names=list(X2.columns))
tree2_rules = export_text(dt2, feature_names=list(X2.columns))
tree3_rules = export_text(dt3, feature_names=list(X2.columns))

print(tree1_rules,'\n', tree2_rules,'\n', tree3_rules)


# In[ ]:




