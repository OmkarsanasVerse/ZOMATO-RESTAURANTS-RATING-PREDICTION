from cProfile import label
from os import umask
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
#Load dataset
data = pd.read_csv('zomato.csv')
data
#check data shape
data.shape
#check data columns
data.columns

data.info()
#check missing values
data.isnull().sum()
#dropped 'url' and 'phone' columns
df = data.drop(['url', 'phone'], axis = 1) 
df.head()

#handling missing values
df.dropna(inplace = True) 
df.isnull().sum()

#handling duplicate values
df.duplicated().sum()
df.drop_duplicates(inplace = True)
df.duplicated().sum()

#rename columns properly
df = df.rename(columns = {'approx_cost(for two people)':'cost',
'listed_in(type)':'type', 
'listed_in(city)': 'city'})
df.head()

#cleaning cost column
df['cost'].unique()
df['cost'] = df['cost'].apply(lambda x:x.replace(',', '')) 
df['cost'] = df['cost'].astype(float)
df['cost'].unique()

#handling rate columns
df['rate'].unique()
df = df.loc[df.rate != 'NEW'] # geting rid of 'NEW'
df['rate'].unique()
df['rate'] = df['rate'].apply(lambda x:x.replace('/5', ''))
df['rate'].unique()
df['rate'] = df['rate'].apply(lambda x: float(x))
df['rate']

#Data visualisation
#1
plt.figure(figsize = (17,10))
chains = df['name'].value_counts()[:20]
sns.barplot(x = chains, y=  chains.index,  palette= 'deep')
plt.title('Most famous restaurants chains in bangalore')
plt.xlabel('Number of outlets')
plt.show()

#check online order or not
#2
v = df['online_order'].value_counts()
fig = plt.gcf()
fig.set_size_inches((10,6))
cmap = plt.get_cmap('Set3')
color = cmap(np.arange(len(v)))
plt.pie(v, labels = v.index, wedgeprops= dict(width = 0.6),autopct = '%0.02f', shadow = True, colors=  color)
plt.title('Online orders', fontsize = 20)
plt.show()

#Book table or not
#3
v = df['book_table'].value_counts()
fig = plt.gcf()
fig.set_size_inches((8,6))
cmap = plt.get_cmap('Set1')
color = cmap(np.arange(len(v)))
plt.pie(v, labels = v.index, wedgeprops= dict(width = 0.6),autopct = '%0.02f', shadow = True, colors=  color)
plt.title('Book Table', fontsize = 20)
plt.show()

#Rating Distribution
#4
plt.figure(figsize = (9,7))
sns.distplot(df['rate'])
plt.title('Rating Distribution')
plt.show()

#Most liked dishes
#5
import re
df.index=range(df.shape[0])
likes=[]
for i in range(df.shape[0]):
    array_split=re.split(',',df['dish_liked'][i])
    for item in array_split:
        likes.append(item)
favourite_food = pd.Series(likes).value_counts()
favourite_food.head(30)
cmap = plt.get_cmap('Set3')
color = cmap(np.arange(len(v)))
ax = favourite_food.nlargest(n = 20, keep = 'first').plot(kind = 'bar', figsize = (18,10), title=  'Top 20 Favourite Food counts', color =  color)
for i in ax.patches:
    ax.annotate(str(i.get_height()), (i.get_x() * 1.005, i.get_height() * 1.005))
plt.show()
    
#Most popular cuisines of bangalore
#6
v = df['cuisines'].value_counts()[:15]
plt.figure(figsize = (20,8))
ax  = sns.barplot(x = v.index, y = v, palette = 'Paired')
for i in ax.patches:
    ax.annotate(i.get_height().astype(int), (i.get_x()*1.005, i.get_height()*1.005))
plt.title('Most popular cuisines of Bangalore', fontsize = 20)
plt.xlabel('Cuisines', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.xticks(rotation =90)
plt.show()

#Distribution of cost of food for two people
#7
plt.figure(figsize=(20,8))
sns.distplot(df['cost'])
plt.show()

#Types of Services
#8
ax  = sns.countplot(df['type']).set_xticklabels(sns.countplot(df['type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.title('Type of Service')
plt.show()
name_grp = df.groupby('name')

# Here I selected 20 restaurant based on high votes
#9
v = name_grp['votes'].agg(np.sum).sort_values(ascending = False)[:20]  
plt.figure(figsize = (20,10))
ax = sns.barplot(y = v, x = v.index)
for i in ax.patches:
    ax.annotate(i.get_height().astype(int), (i.get_x()* 1.005, i.get_height()*1.005))
plt.title('Highest vote of restaurant', fontsize = 20)
plt.xlabel('Restaurant', fontsize = 15)
plt.ylabel('Frequecy', fontsize = 15)
plt.xticks(rotation =90)
plt.show()
df.head()

#Convert the online categorical variables into numeric format
#10
df.online_order[df.online_order == 'Yes'] = 1
df.online_order[df.online_order == 'No'] =  0
df.online_order.value_counts()
#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.location = le.fit_transform(df.location)
df.rest_type = le.fit_transform(df.rest_type)
df.cuisines = le.fit_transform(df.cuisines)
df.menu_item = le.fit_transform(df.menu_item)
df.book_table = le.fit_transform(df.book_table)
df.head(n=2)
my_data = df.iloc[:,[2,3,4,5,6,7,9,10,12]]
my_data.to_csv('Zomato_df.csv')
my_data.head()
plt.figure(figsize = (20,20))
sns.heatmap(my_data.corr(), annot = True)
plt.show()

#Dependent and independent variables
X = df.iloc[:,[2,3,5,6,7,9,10,12]]
y = df['rate']
X
y

#Spliting data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=10)
X_train
y_train

#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
#predict the test set
y_pred = lr.predict(X_test)
## Evaluate the model
from sklearn.metrics import r2_score
print('Linear Tree Regression : ',r2_score(y_test, y_pred))

#Decision tree regression
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_leaf=0.01)
dtr.fit(X_train,y_train)
# Predict the test ser
y_pred  = dtr.predict(X_test)
# Evaluate the model performance
from sklearn.metrics import r2_score
print('Decision Tree Regression : ',r2_score(y_test,y_pred))

#Random Forest regression
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=650,random_state=245,min_samples_leaf=.0001)
rfr.fit(X_train,y_train)
# Predict the test ser
y_pred  = rfr.predict(X_test)
# Evaluate the model performance
from sklearn.metrics import r2_score
print('Random Forest Regression : ',r2_score(y_test,y_pred))

#Support vector Regression
from sklearn.svm import SVR
svr = SVR(kernel ='rbf')
svr.fit(X_train, y_train)
# predict the test set
y_pred = svr.predict(X_test)
# Evaluate the performance
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print('Support Vector Regression : ',r2)

#Extra trees regressor
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor(n_estimators = 120)
etr.fit(X_train, y_train)
# predict the test set
y_pred = etr.predict(X_test)
# Evaluate the model performance
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print('Extra tree regressor : ',r2) 