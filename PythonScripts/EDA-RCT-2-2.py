
# coding: utf-8

# # Kickstarter Projects 
# ### CSC 478 Final Project
# #### Synopsis:
# * The purpose of this project is to predict whether a kickstarter campaign will fail, succeed, or cancel based on the available information available [here](https://raw.githubusercontent.com/stfox13/CSC478FinalProject/master/Data/ks-projects-201801.csv).
# * We will use an array of machine learning algorithms, including KNN, Linear Regression, Logistic Regression, and / or SVM to find the most accurate model.
# 
# #### Contributors:
# * [Rebecca Tung (1448196)](https://github.com/rtungus)
# * [Sidney Fox (1524992)](https://github.com/stfox13)
# 
# #### Data Dictionary:
# 
# 
# ##### Content
# You'll find most useful data for project analysis. Columns are self explanatory except:
# 
# 1. usd_pledged: conversion in US dollars of the pledged column (conversion done by kickstarter).
# 
# 2. usd pledge real: conversion in US dollars of the pledged column (conversion from Fixer.io API).
# 
# 3. usd goal real: conversion in US dollars of the goal column (conversion from Fixer.io API).
# 
# 
# |Sequence Number|Column Name|Data Type|
# |:---|:---|:---|
# |1|ID|Numeric|
# |2|name|String|
# |3|category|String|
# |4|main_category|String|
# |5|currency|String|
# |6|deadline|DateTime|
# |7|goal|Numeric|
# |8|launched|DateTime|
# |9|pledged|Numeric|
# |10|state|String|
# |11|backers|Numeric|
# |12|country|String|
# |13|usd pledged|Numeric|
# |14|usd_pledged_real|Numeric|
# |15|usd_goal_real|Numeric|

# ## Libraries used through the project:

# In[1]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
#import plotly.plotly as py
import numpy as np
import pandas as pd
import seaborn as sns
import os
import math
import requests
import datetime as dt
import matplotlib as mpl
import io
from pandas import Series, DataFrame
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import itertools
from sklearn.feature_selection import RFE
from collections import defaultdict


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


#Set graph size
mpl.rcParams['figure.figsize'] = (12,12)


# In[4]:


np.set_printoptions(suppress=True)


# In[5]:


def roundup(x, y):
    #return int(math.ceil(x / float(y))) * y
    return int(math.ceil(x / y) * y)


# ## Load raw data as Pandas DataFrame:

# In[6]:


url = 'https://raw.githubusercontent.com/stfox13/CSC478FinalProject/master/Data/ks-projects-201801.csv'
#url='ks-projects-201801.csv'
kickproj_org= pd.read_csv(url)
len(kickproj_org)


# ## Define Reuseful Function

# In[7]:


def roundup(x, y):
    #return int(math.ceil(x / float(y))) * y
    return int(math.ceil(float(x) / float(y)) * y)


# In[8]:


#Define a fuction to print and plot confusin matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass
#   print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[9]:


#Define a fuction to calculate and print TP, TN, FP, and FN for each category
def show_statistics(test_y, y_pred, matrix):
    TP = np.diag(matrix)
    FP = np.sum(matrix, axis=0) - TP
    FN = np.sum(matrix, axis=1) - TP
    TN = []
    for i in range(len(matrix)):
        temp = np.delete(matrix, i, 0)    # delete ith row
        temp = np.delete(temp, i, 1)  # delete ith column
        TN.append(sum(sum(temp)))    
    temp_dic = {'TP': TP, 'FP' : FP, 
                'TN' : TN, 'FN' : FN}
    scoreMatrix = DataFrame.from_dict(temp_dic)
    #print "TP, TN, FP, FN for each cateory: "
    return scoreMatrix


# In[10]:


# Define a fuction to print F1 Score for each class and global (micro)
def formatResult(preResult, columnNames):  
    resultDF = DataFrame(preResult.values(), columns=columnNames, index=preResult.keys())
    resultDF.loc['sum'] = np.sum(preResult.values(), axis=0)
    resultDF['Sum of Class F1'] = np.append(np.sum(preResult.values(), axis=1), np.NaN)
    return resultDF


# In[11]:


#######KNN############
#Define a function to run KNeighborsClassifier with different n_neighbors and store f1 score
def runKNN(trainX, trainY, testX, testY, number, f1_only = False, trainSetName = '', dic_result_knn = {}):
    i = 3
    cls = KNeighborsClassifier(n_neighbors=i)
    while i <= number:
        #print i
        cls = KNeighborsClassifier(n_neighbors=i)
        cls.fit(trainX, trainY)
        predY = cls.predict(testX)
        result = f1_score(testY, predY, average=None).round(2)
        result = np.append(result, f1_score(testY, predY, average='micro').round(2))
        
        #print results
        dic_result_knn['N=' + str(i) +'-' + trainSetName] = result
        #print "n_neighbors = " + str(i) + " : " + result
        i = i + 2
    return dic_result_knn


# In[12]:


#######LogisticRegression############
# Define a function to run LogisticRegression with different class_weight settings and store f1 score
def runLogistic(trainX, trainY, testX, testY, f1_only = False, trainSetName = '', dic_result_log = {}):
    cls = LogisticRegression() 
    cls.fit(trainX, trainY)
    predY = cls.predict(testX)
    result = f1_score(testY, predY, average=None).round(2)
    result = np.append(result, f1_score(testY, predY, average='micro').round(2))
    #print results
    dic_result_log['CWeight = None - ' + trainSetName] = result 
    
    cls = LogisticRegression(class_weight='balanced')
    cls.fit(trainX, trainY)
    predY = cls.predict(testX)
    result = f1_score(testY, predY, average=None).round(2)
    result = np.append(result, f1_score(testY, predY, average='micro').round(2))
    #print results
    dic_result_log['CWeight = balanced - ' + trainSetName] = result   
    return dic_result_log


# In[13]:


#######SVM############
# Define a function to run SVM with different kernel settings and store f1 score
def runSVM(trainX, trainY, testX, testY, f1_only = False, trainSetName = '', dic_result_log = {}):
    
    C = 1.0 # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(trainX, trainY)
    predY = svc.predict(X_plot)        
    result = f1_score(testY, predY, average=None).round(2)
    result = np.append(result, f1_score(testY, predY, average='micro').round(2))
    print results
    dic_result_log['SVCKernel = linear - ' + trainSetName] = result 
    
    svc = svm.SVC(kernel='rbf', C=C, decision_function_shape='ovr').fit(trainX, trainY)
    predY = svc.predict(X_plot)        
    result = f1_score(testY, predY, average=None).round(2)
    result = np.append(result, f1_score(testY, predY, average='micro').round(2))
    print results
    dic_result_log['SVCKernel = rbf - ' + trainSetName] = result 
    
    #svc = svm.SVC(kernel='poly', C=C, decision_function_shape='ovr').fit(trainX, trainY)
    #predY = svc.predict(X_plot)        
    #result = f1_score(testY, predY, average=None).round(2)
    #result = np.append(result, f1_score(testY, predY, average='micro').round(2))
    #print results
    #dic_result_log['SVCKernel = poly - ' + trainSetName] = result 
    
    return dic_result_log


# ## Check the Y data:

# In[14]:


#Plot histogram
kickproj_org['state'].value_counts().plot(kind='bar', title='Project State Histograms')


# ### Drop projects when the state is equal to "undefined":

# In[15]:


# Remove state = 'undefined'
kickproj = kickproj_org[(kickproj_org['state'] != 'undefined') & (kickproj_org['state'] != 'live')]
len(kickproj)
kickproj['state'].value_counts().plot(kind='bar', title='Project State Histograms')


# In[16]:


kickproj.head(5)


# #### Since we have the goal and pledge amounts converted to US dollars (usd), we will drop the original goal and pledged columns:

# In[17]:


#kickproj = kickproj[kickproj.columns.difference(['goal','pledged','usd pledged'])]
kickproj = kickproj.drop(['goal','pledged','usd pledged'], axis=1)
len(kickproj)
kickproj.head(5)


# ## Check the X data:

# In[18]:


kickproj.describe()


# In[19]:


print('Heat Map of Correlation Coefficients:')
#sns.heatmap(kickproj.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).round(3), cmap=sns.diverging_palette(10, 220, sep=80, n=7), linewidths=0.1, annot=True, vmin=-1, vmax=1)
sns.heatmap(kickproj.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).round(3), cmap='RdYlBu', linewidths=0.1, annot=True, vmin=-1, vmax=1)


# In[20]:


categoryDF = kickproj.groupby(['category']).size().reset_index(name='counts')
len(categoryDF)
categoryDF.head(5)


# In[21]:


kickproj.groupby(['main_category']).size().reset_index(name='counts')
kickproj['main_category'].value_counts().plot(kind='bar', title='Project Main Category Histograms')


# In[22]:


cateDF = kickproj.groupby(['main_category', 'category']).size().reset_index(name='counts')
len(cateDF)
cateDF.head(40)


# In[23]:


kickproj.groupby(['country']).size().reset_index(name='counts')
kickproj['country'].value_counts().plot(kind='bar', title='Project Country Histograms')


# ### Remove country with invalid value, N,0"

# In[24]:


kickproj = kickproj[kickproj['country'] != 'N,0"']
kickproj.groupby(['country']).size().reset_index(name='counts')
kickproj['country'].value_counts().plot(kind='bar', title='Project Country Histograms')


# ### Check null value 

# In[25]:


null_columns=kickproj.columns[kickproj.isnull().any()]
null_columns
kickproj[null_columns].isnull().sum()
kickproj[kickproj["name"].isnull()][null_columns] 


# ### Replace nan with Unknow for name

# In[26]:


kickproj["name"].fillna('Unknown', inplace=True)
null_columns=kickproj.columns[kickproj.isnull().any()]
null_columns


# ### Apply correct data types to DataFrame:

# In[27]:


print 'Data types do not align with the data types defined in the data dictionary:\n\n', kickproj.dtypes


# In[28]:


# Columns that are of date data type:
datecols = ['deadline','launched']
# Columns that are of int data type:
intcols = ['usd_pledged_real','usd_goal_real']

for col in datecols:
    kickproj[col] = pd.to_datetime(kickproj[col])
    kickproj[col] = [d.date().toordinal() for d in kickproj[col]]

kickproj[intcols] = kickproj[intcols].fillna(0).astype(np.int64)
kickproj['duration'] = abs(kickproj['deadline']-kickproj['launched'])


# In[29]:


print 'Review converted data types:\n\n', kickproj.dtypes


# ### Check the range of usd_pledged_real and usd_goal_real

# In[30]:


binrange = range(1, roundup(max(kickproj['usd_pledged_real']),100000), 5000000)
binrange


# In[31]:


min(kickproj['usd_goal_real'])
max(kickproj['usd_goal_real'])


# In[32]:


min(kickproj['usd_pledged_real'])
max(kickproj['usd_pledged_real'])


# ### Check whether All successful records have usd_pledged_real > 0 - Outliners to be removed

# In[33]:


# All successful records have usd_pledged_real > 0? - There is one record with excpetion and we remove it
min(kickproj['usd_pledged_real'])
x = kickproj[(kickproj['usd_pledged_real']==0) & (kickproj['state']=='successful')].index
kickproj.drop(x, inplace=True)
kickproj[(kickproj['usd_pledged_real']==0) & (kickproj['state']=='successful')]


# ### Find out correlation among variables
# 1. ** Feature X - backer (0.33), duration (0.11), usd_pledged_real(0.45), usd_goal_real (-0.07), currency (-0.05) and country (-0.04) are strongly correlated with State (Target Variable) **
# 2. ** Currency and Country are highly correlated (0.94).  Only one should be used in the model.  We decide to go with Country **
# 3. ** Duration is derived from deadline and launched.  Duration will be used instead of deadline and launched in the model.**

# In[34]:


print('Heat Map of Correlation Coefficients:')
#sns.heatmap(kickproj.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).round(3), cmap=sns.diverging_palette(10, 220, sep=80, n=7), linewidths=0.1, annot=True, vmin=-1, vmax=1)
sns.heatmap(kickproj.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).round(3), cmap='RdYlBu', linewidths=0.1, annot=True, vmin=-1, vmax=1)


# #### Create Projects Summary DataFrame:

# kickproj_summary = kickproj.groupby(['main_category','state'], as_index=False) \
# [['usd_goal_real','usd_pledged_real','backers']].agg({
#     'usd_goal_real':[np.sum, np.mean, np.std],
#     'usd_pledged_real':[np.sum, np.mean, np.std],
#     'backers':[np.sum, np.mean, np.std,'count']
# })
# 
# #"Flatten" summary results:
# kickproj_summary.columns = list(map('_'.join, kickproj_summary.columns.values))
# 
# #Rename a few columns:
# kickproj_summary.columns.values[-1] = 'project_count'
# kickproj_summary.columns.values[0] = 'main_category'
# kickproj_summary.columns.values[1] = 'state'
# 
# kickproj_summary.head()

# #### Data Visualization
# 
# from ggplot import *
# 
# ggplot(kickproj, aes(x='backers', y='usd_pledged_real', color='state')) +\
#     geom_point(size=20) +\
#     xlab('backers') + ylab('usd_pledged_real') + ggtitle('Backers v. Pledged\nFaceted by Project State') +\
#     scale_x_continuous(limits=(0,max(kickproj.backers))) +\
#     scale_y_continuous(limits=(0,max(kickproj.usd_pledged_real))) +\
#     theme_bw() +\
#     facet_grid('state')    
# 
# fig=plt.figure(figsize=(16, 16), dpi= 100, facecolor='w', edgecolor='k')
# 
# ggplot(kickproj[(kickproj['state'] =='successful')], aes(x='backers', y='usd_pledged_real', color='state')) +\
#     geom_point(size=20) +\
#     xlab('backers') + ylab('usd_pledged_real') + ggtitle('Backers v. Pledged\nState is equal to Successful') +\
#     scale_x_continuous(limits=(0,150000),labels='comma') +\
#     scale_y_continuous(limits=(0,max(kickproj.usd_pledged_real)),labels='comma') +\
#     theme_bw() 

# ### Shuffle the dataset and create training and test datasets

# In[35]:


shffled_kickproj = kickproj.sample(frac=1)


# #### Convert the value of state to True (success) or False (Other)

# In[36]:


shffled_kickproj['state_cd'] = shffled_kickproj['state'].apply(lambda a: True if a == 'successful' else False)
shffled_kickproj.head(5)


# #### Convert each country to a number

# In[37]:


le = preprocessing.LabelEncoder()
le.fit(shffled_kickproj['country'])
shffled_kickproj['country_cd'] = le.transform(shffled_kickproj['country'])
shffled_kickproj.head(5)


# In[38]:


num = shffled_kickproj.shape[0]/2

fi_vals, train_x, train_y = shffled_kickproj.iloc[0:num, [1,2,3,4,5,6,7,8,9,10,11,12,13,14]], shffled_kickproj.iloc[0:num, [8,9,10,11,12,14]], shffled_kickproj.iloc[0:num, 13]
test_x, test_y = shffled_kickproj.iloc[num:, [8,9,10,11,12,14]], shffled_kickproj.iloc[num:, 13]
train_x.head(2)
train_x.shape
train_y.head(2)
train_y.shape
test_x.head(2)
test_x.shape
test_y.head(2)
test_y.shape


# ### Train training set
# #### Convert country using oneHotEncoder

# In[39]:


temp_features_train = train_x['country_cd'].reshape(-1, 1) # Needs to be the correct shape
temp_features_test = test_x['country_cd'].reshape(-1, 1) # Needs to be the correct shape


# In[40]:


ohe = preprocessing.OneHotEncoder(sparse=False) #Easier to read
#fit on training set only
ohe.fit(temp_features_train)
countryDF_train = DataFrame(ohe.transform(temp_features_train), columns = ohe.active_features_, index = train_x.index)
countryDF_test = DataFrame(ohe.transform(temp_features_test), columns = ohe.active_features_, index = test_x.index)
countryDF_train.head(10)
countryDF_test.head(10)


# In[41]:


train_x.shape
countryDF_train.shape
test_x.shape
countryDF_test.shape


# In[42]:


train_X1 = pd.merge(train_x.drop(['country','country_cd'], axis=1), countryDF_train, left_index=True, right_index=True)
train_X1.head(10)
train_X1.shape

test_X1 = pd.merge(test_x.drop(['country','country_cd'], axis=1), countryDF_test, left_index=True, right_index=True)
test_X1.head(10)
test_X1.shape


# In[43]:


result_Dic = {}
#Call function to run KNeighborsClassifier with different n_neighbors settings (up to 20) and store the f1 score results
result_Dic = runKNN(train_X1, train_y.values.ravel(), test_X1, test_y.values.ravel(), 18, trainSetName = 'Basic&Country', dic_result_knn = result_Dic)


# In[44]:


#Call function to run LogisticRegression with different class_weight settings (None or Balance)  and store the f1 score results
result_Dic = runLogistic(train_X1, train_y.values.ravel(), test_X1, test_y.values.ravel(), trainSetName = 'Basic&Country', dic_result_log = result_Dic)


# In[45]:


# Call function to run LogisticSVM with different kernel settings (linear, rbf or poly)  and store the f1 score results
# result_Dic = runSVM(train_X1, train_y.values.ravel(), test_X1, test_y.values.ravel(), trainSetName = 'Basic&Country', dic_result_log = result_Dic)


# In[46]:


#Store number of classes
n_classes = np.unique(shffled_kickproj['state_cd'])
n_classes
resultDF = formatResult(result_Dic, np.append(n_classes, 'micro'))
resultDF.head(10)


# In[47]:


# Print out the top 10 Micro (overall) F1 score from all settings
resultDF.drop('sum').nlargest(5, 'micro')
resultDF.drop('sum').nlargest(5, 'Sum of Class F1')


# In[48]:


knnClr = KNeighborsClassifier(n_neighbors=5)
knnClr.fit(train_X1, train_y.values.ravel())
final_y_pred = knnClr.predict(test_X1)


# In[49]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y, final_y_pred)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[False,True],
                      normalize=False,
                      title='Confusion matrix, without normalization (True = successful; False = Other)')


# In[50]:


stats = show_statistics(test_y, final_y_pred, cnf_matrix)
strName = map((lambda a: 'successful' if a == True else 'Failed/Cancelled/Live/Suspended'),  [False, True])
print "Classificaiton Reprt:"
print classification_report(test_y, final_y_pred, target_names=strName, digits=2)


# # Feature Importance
# 

# In[65]:


#Encode non-numeric variables - needed to run most of the models, understand anything feature importance:
le = preprocessing.LabelEncoder
d = defaultdict(le)
le_df = fi_vals.drop(
    ['currency','deadline','launched','country','state'], axis=1).apply(
    lambda x: d[x.name].fit_transform(x))


# In[66]:


le_features, le_target = le_df[le_df.columns.drop('state_cd')], le_df['state_cd']


# In[67]:


#We'll look at recursive feature elimination   (RFE) with logistic regression and select three features:
LogReg_RFE = RFE(LogisticRegression(), 3).fit(le_features, le_target)
print('The three most important features according to Logistic Regression:\n'),(np.array(le_features.columns)[LogReg_RFE.support_])


# In[68]:


#We'll look at recursive feature elimination (RFE) with linear regression and select three features:
LinReg_RFE = RFE(LinearRegression(), 3).fit(le_features, le_target)
print('The three most important features according to Linear Regression:\n'),(np.array(le_features.columns)[LinReg_RFE.support_])


# In[78]:


#We'll use extra trees classifier to calculate feature importance:
ETC = ExtraTreesClassifier().fit(le_features, le_target)
feat_imp_df = pd.DataFrame({'Columns':pd.Series(le_features.columns)})
feat_imp_df['Feature Importance'] = pd.Series(ETC.feature_importances_)
feat_imp_df.set_index(['Columns'],inplace=True)
print('Column Names and Associated Feature Importance:')
feat_imp_df
feat_imp_df.plot(kind="bar", title="Feature Importance Results\nTraining non-OHE Data", legend = False)


# In[70]:


temp_features_fi = fi_vals['country_cd'].reshape(-1, 1) # Needs to be the correct shape
ohe = preprocessing.OneHotEncoder(sparse=False) #Easier to read
ohe.fit(temp_features_fi)
countryDF_fi = DataFrame(ohe.transform(temp_features_fi), 
                            columns = ohe.active_features_, index = fi_vals.index)


# In[71]:


fi_vals_ohe = pd.merge(fi_vals.drop(['country','country_cd'], axis=1), countryDF_fi, left_index=True, right_index=True)
fi_vals_ohe


# In[72]:


ohe_df = fi_vals_ohe.drop(
    ['currency','deadline','launched','state'], axis=1).apply(
    lambda x: d[x.name].fit_transform(x))


# In[73]:


ohe_features, ohe_target = ohe_df[ohe_df.columns.drop('state_cd')], ohe_df['state_cd']


# In[74]:


#We'll look at recursive feature elimination   (RFE) with logistic regression and select three features:
LogReg_RFE = RFE(LogisticRegression(), 3).fit(ohe_features, ohe_target)
print('The three most important features according to Logistic Regression:\n'),(np.array(ohe_features.columns)[LogReg_RFE.support_])


# In[75]:


#We'll look at recursive feature elimination (RFE) with linear regression and select three features:
LinReg_RFE = RFE(LinearRegression(), 3).fit(ohe_features, ohe_target)
print('The three most important features according to Linear Regression:\n'),(np.array(ohe_features.columns)[LinReg_RFE.support_])


# In[76]:


np.unique(fi_vals[['country','country_cd']].values)


# In[79]:


#We'll use extra trees classifier to calculate feature importance:
ETC = ExtraTreesClassifier().fit(ohe_features, ohe_target)
feat_imp_df = pd.DataFrame({'Columns':pd.Series(ohe_features.columns)})
feat_imp_df['Feature Importance'] = pd.Series(ETC.feature_importances_)
feat_imp_df.set_index(['Columns'],inplace=True)
print('Column Names and Associated Feature Importance:')
feat_imp_df.sort_values(by=['Feature Importance'])
feat_imp_df.plot(kind="bar", title="Feature Importance Results\nTraining OHE Data", legend = False)

