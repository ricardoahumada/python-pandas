#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mysql.connector as connection
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

try:
    mydb = connection.connect(host="db4free.net", database = 'dbtraining',user="dbuseracc1", passwd="Pa55w0rd",use_pure=True)
    query = "Select * from customers;"
    result_dataFrame = pd.read_sql(query,mydb)
    mydb.close() #close the connection
except Exception as e:
    mydb.close()
    print(str(e))


# In[3]:


result_dataFrame


# In[4]:


result_dataFrame.to_csv('data/import_customers_2.csv')

