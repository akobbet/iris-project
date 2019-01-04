
# coding: utf-8

# # Import librairies

# In[ ]:


# %load multi_class_libraries
# Load libraries
import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# # Load Dataset
# I am going to load the dataset from my computer. The dataset is from UCI Machine Learning repo

# In[18]:


filename = 'iris.csv'


# In[21]:


# read iris.csv into DataFrame called filename
df = pd.read_csv(filename)


# # Summary of the dataset

# In[23]:


# shape of the dataset
df.shape


# In[26]:


# type
df.dtypes


# In[48]:


# view the first 5 rows of the dataset
df.head()


# In[28]:


# statistical summary
df.describe()


# In[29]:


# class distribution
df.groupby('species').size()


# # Data Visualization

# In[31]:


# histogram plots
df.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
pyplot.show()


# In[32]:


# box & whisker plots
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[33]:


# scatter plots matrix
scatter_matrix(df)
pyplot.show()


# In[36]:


#split data into train and validation datasets
array = df.values
X = array[:,0:4]
Y= array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[37]:


# test options and evaluation metric
num_folds = 10
seed = 7
scoring ='accuracy'


# In[38]:


# Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[41]:


# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[42]:


# Visualization of the Distribution of Algorithms Performance
# compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


# # Finalize Model

# In[44]:


# make prediction on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# I can see that the accuracy is 0.9 or 90%. The confusin matrix provides an indication of the three errors made. Finally the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results(granted the validation dataset was small).
