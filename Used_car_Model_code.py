#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


cars_df= pd.read_csv(r"C:\Users\SHUBHAM WAGHMARE\Downloads\ML_project_learning\Used_cars_dataset\cardekho_imputated.csv")


# In[5]:


cars_df.drop(cars_df.columns[[0, 2, 3]], axis = 1, inplace = True)


# In[6]:


cars_df.dtypes


# In[7]:


cars_df['fuel_type'].unique()


# In[8]:


cars_df['seller_type'].value_counts()


# In[9]:


cars_df['seats'].value_counts()


# In[10]:


cars_df[cars_df['seats']==0]


# In[11]:


seats={0:5}
cars_df['seats'].replace(seats, inplace=True)


# In[12]:


cars_df.isna().sum()


# In[13]:


cars_df.info()


# In[14]:


cars_df.describe()


# # Gaining_insights

# In[15]:


import matplotlib.pyplot as plt


# In[16]:


cars_df.hist(bins=30, figsize=(20,15))


# In[17]:


corr_matrix = cars_df.corr()


# In[18]:


corr_matrix['selling_price'].sort_values(ascending=False)


# In[19]:


from pandas.plotting import scatter_matrix

attributes1= ['selling_price','max_power','max_cost_price','engine','min_cost_price']
attributes2=['selling_price','seats','vehicle_age','mileage']
scatter_matrix(cars_df[attributes1], figsize=(12,8))
scatter_matrix(cars_df[attributes2], figsize=(12,8))


# In[20]:


# checking more on high corr of max_power and selling price


# In[21]:


plt.figure(figsize=(10,5))
plt.scatter(cars_df['max_power'],cars_df['selling_price'],alpha=0.1)   
plt.show()
#cars_df.plot(kind="scatter", x="max_power", y="selling_price",alpha=0.1)


# In[22]:


# similarly for mileage
plt.figure(figsize=(5,3))
plt.scatter(cars_df['mileage'],cars_df['selling_price'],alpha=0.1)   
plt.show()


# In[23]:


#combining min_price and max_price column as avg_price


# In[24]:


cars_df['avg_cost_price'] =cars_df[['min_cost_price','max_cost_price']].mean(axis=1)


# In[25]:


cars_df.drop(cars_df.columns[[0,1,2]],axis=1,inplace=True)


# In[26]:


cars_df.head()


# In[27]:


#removing limit values


# In[28]:


cars_df.drop(cars_df[(cars_df['vehicle_age'] > 20) ].index, inplace = True)
cars_df.drop(cars_df[cars_df['km_driven'] >300000 ].index, inplace = True)


# In[29]:


def removeOutliers(data, col):
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)
    IQR = Q3 - Q1
      
    print("IQR value for column %s is: %s" % (col, IQR))
    global outlier_free_list
    global filtered_data
      
    lower_range = Q1 - 1.5 * IQR
    upper_range = Q3 + 1.5 * IQR
    outlier_free_list = [x for x in data[col] if (
        (x > lower_range) & (x < upper_range))]
    filtered_data = data.loc[data[col].isin(outlier_free_list)]

out_columns = cars_df[['km_driven','vehicle_age','mileage','engine','max_power','seats','selling_price']]
for i in out_columns:
    removeOutliers(cars_df, i)
cars_df = filtered_data
print("Shape of data after outlier removal is: ", cars_df.shape)


# In[30]:


cars_df.info()


# # Handling categorical values

# In[31]:


cars_df=pd.get_dummies(cars_df,columns=['fuel_type','transmission_type','seller_type'],drop_first=True)
cars_df.head()


# # Splitting data for training and validation

# In[32]:


from sklearn.model_selection import train_test_split

X=cars_df
y=cars_df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[33]:


print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# # Model_Selection

# In[34]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
#cross_val = cross_val_score(model ,X_train ,y_train ,cv=3)
#cross_val_mean = cross_val.mean()


# In[35]:


from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
cross_val1= cross_val_score(tree_reg ,X_train ,y_train ,cv=5)
cross_val_mean = cross_val1.mean()


# In[36]:


cross_val1


# # linear_regressor

# In[37]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
cross_val2= cross_val_score(lin_reg ,X_train ,y_train ,cv=5)
cross_val_mean = cross_val2.mean()
print(cross_val2)
print(cross_val_mean)


# In[38]:


lin_reg = LinearRegression()
lin_reg.fit(X_train ,y_train)
test_pred=lin_reg.predict(X_test)
#print(mean_absolute_error(X_test,y_test))
test_pred


# In[39]:


print(mean_absolute_error(test_pred,y_test))


# In[40]:


df=pd.DataFrame(y_test)
df['pred']=test_pred
df['diff']=df.pred-df.selling_price
df['diff'].mean()


# # random_forest

# In[52]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
cross_val3= cross_val_score(forest_reg ,X_train ,y_train ,cv=5)
cross_val_mean = cross_val3.mean()
print(cross_val3)
print(cross_val_mean)


# In[ ]:




