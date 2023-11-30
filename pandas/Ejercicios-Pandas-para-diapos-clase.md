# Basic
import pandas as pd

data = { 'apples':[3, 2, 0, 1] , 'oranges':[0, 3, 7, 2] }

df = pd.DataFrame(data)

df

df = pd.DataFrame(data, index=['Luis', 'Ana', 'Juana', 'Pedro'])

df

df.loc['Ana']

# ============================================================================

# Crear Dataframes

## pandas.DataFrame.from_dict

data = {'col_1':[3, 2, 1, 0], 'col_2':['a','b','c','d']}
pd.DataFrame.from_dict(data)

data = {'row_1':[3, 2, 1, 0], 'row_2':['a','b','c','d']} 
pd.DataFrame.from_dict(data, orient='index')

## Leer datos de un CSV
df = pd.read_csv('data/frutas.csv')
df

dff = pd.read_csv('data/frutas.csv', index_col=0)
dff

## Leer datos de un JSON
df = pd.read_json('data/frutas.json')
df

df = pd.read_json('data/dataset.json')
df

df = pd.read_json('data/dataset.json', orient='column')
df

df = pd.read_json('data/dataset.json', orient='index')
df

## Convertir a CSV o JSON
df.to_csv('data/new_dataset.csv') 
df.to_json('data/new_dataset.json') 

# ============================================================================
# Operaciones más importantes de DataFrame
movies_df = pd.read_csv("data/movies/movies.csv", index_col="title")

movies_df.head()
movies_df.tail(2)

movies_df.info()
movies_df.shape

## Duplicados
temp_df = pd.concat([movies_df, movies_df], ignore_index=True)
temp_df.shape

temp_df = temp_df.drop_duplicates() 
temp_df.shape

temp_df = pd.concat([movies_df, movies_df], ignore_index=True)
temp_df.shape

temp_df.drop_duplicates(inplace=True)
temp_df.shape

## Entender las variables
movies_df.describe()

movies_df['genres'].describe()

# ============================================================================
# Más ejemplos

## 1
data = [1,2,3,10,20,30]
df = pd.DataFrame(data)
print(df)


data = {'Name' : ['AA', 'BB'], 'Age': [30,45]}
df = pd.DataFrame(data)
print(df)

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print(df)

data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
print(df)

## 2
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

#With two column indices, values same as dictionary keys
df1 = pd.DataFrame(data,index=['first','second'],columns=['a','b'])

#With two column indices with one index with other name
df2 = pd.DataFrame(data,index=['first','second'],columns=['a','b1'])

print(df1)
print('...........')
print(df2)

## 3
d = {'one' : pd.Series([1, 2, 3]  , index=['a', 'b', 'c']), 
     'two' : pd.Series([1,2, 3, 4], index=['a', 'b', 'c', 'd'])
    }
df = pd.DataFrame(d)
print(df)

## 4
d = {'one':pd.Series([1,2,3],   index=['a','b','c']), 
     'two':pd.Series([1,2,3,4], index=['a','b','c','d'])
    }
df = pd.DataFrame(d)
# Adding a new column to an existing DataFrame object
# with column label by passing new series

print("Adding a new column by passing as Series:")
df['three'] = pd.Series([10,20,30],index=['a','b','c'])
print(df)

print("Adding a column using an existing columns in DataFrame:")
df['four'] = df['one']+df['three']
print(df)


## 5
d = {'one'   : pd.Series([1, 2, 3],    index=['a', 'b', 'c']),
     'two'   : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']),
     'three' : pd.Series([10,20,30],   index=['a','b','c'])
    }
df = pd.DataFrame(d)
print ("Our dataframe is:")
print(df)

# using del function
print("Deleting the first column using DEL function:")
del df['one']
print(df)

# using pop function
print("Deleting another column using POP function:")
df.pop('two')
print(df)

## 6
d = {'one' : pd.Series([1, 2, 3],    index=['a', 'b', 'c']),
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c','d'])
    }
df = pd.DataFrame(d)
print(df[2:4])


## 7
d = {'one' : pd.Series([1, 2, 3],    index=['a', 'b', 'c']),
     'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c','d'])
    }
df = pd.DataFrame(d)
print(df)

df2 = pd.DataFrame([[5,6], [7,8]], columns = ['a', 'b'])
df = df.append(df2 )
print(df)


## 8
d = {'one':pd.Series([1, 2, 3],    index=['a','b','c']),
     'two':pd.Series([1, 2, 3, 4], index=['a','b','c','d'])
    }
df = pd.DataFrame(d)
print(df)

df2 = pd.DataFrame([[5,6], [7,8]], columns = ['a', 'b'])
df = df.append(df2 )
print(df)

df = df.drop(0)
print(df)

## 9
# Creating the first dataframe
df1 = pd.DataFrame({"A":[1, 5, 3, 4, 2],
		     "B":[3, 2, 4, 3, 4],
		     "C":[2, 2, 7, 3, 4],
		     "D":[4, 3, 6, 12, 7]},
		     index =["A1", "A2", "A3", "A4", "A5"])

# Creating the second dataframe
df2 = pd.DataFrame({"A":[10, 11, 7, 8, 5],
		     "B":[21, 5, 32, 4, 6],
		     "C":[11, 21, 23, 7, 9],
		     "D":[1, 5, 3, 8, 6]},
		     index =["A1", "A3", "A4", "A7", "A8"])

# Print the first dataframe
print(df1)
print(df2)
# find matching indexes
df1.reindex_like(df2)


## 10
df1 = pd.DataFrame({'Name':['A','B'], 'SSN':[10,20], 'marks':[90, 95] })
df2 = pd.DataFrame({'Name':['B','C'], 'SSN':[25,30], 'marks':[80, 97] })
df3 = pd.concat([df1, df2])
df3


# ============================================================================
# Manejo de datos categóricos

cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
print(cat)

import numpy as np
cat = pd.Categorical(["a", "c", "c", np.nan], categories=["b", "a", "c"])
df = pd.DataFrame({"cat": cat, "s":["a", "c", "c", np.nan]})
print(df.describe())
print(df["cat"].describe())


# ============================================================================
# Lectura de datos de una base de datos SQL

# !pip install pysqlite3

import sqlite3 
con = sqlite3.connect("data/database.db")


df = pd.read_sql_query("SELECT * FROM movies", con) 
df

df = pd.read_sql_query("SELECT * FROM movies", con,index_col='title')
df

movies_df.to_sql(name='movies', con=con)

# ============================================================================
# Otras funcionalidades interesantes:

df = pd.read_csv('data/iris.data',header=None)

df = pd.read_csv('data/iris.data',header=None)
df

nombres = ['long_sepalo','ancho_sepalo','long_petalo','ancho_petalo','clase']
df.columns = nombres
df

df.index


df['clase'].value_counts()

df.memory_usage()

df.T

df.sort_values('ancho_sepalo',ascending=False)

df[['long_sepalo','long_petalo']]
df[:3]
df.loc[[3,1,5],['ancho_sepalo']]
df.iloc[:5,2:]
df.iloc[[4,19],[0,2]]
df[(df['long_sepalo']>5) & (df['long_petalo']>2)]


df['long_sepalo']-df['long_petalo']

df.isna()
df.isna().sum()
df.isna().sum().sum()

df['long_petalo'][:2]=np.nan
df['long_petalo'].fillna(2)
df

df['long_petalo']=df['long_petalo'].fillna(2)
df

df['long_petalo'].mean()
df['long_petalo'].median()

df_group=df.groupby('clase')['ancho_petalo'].mean()
df_group.name='media_ancho_petalo'
df_group


# ============================================================================
# Entendiendo los datos con estadísticas

df.head()
df.shape
df.dtypes
df.describe()
df.groupby()
df.corr(method='pearson')
df.skew()

# ============================================================================
# Tipos de gráficos
!pip install matplotlib

import pandas as pd
import matplotlib.pyplot as plt

nombres = ['long_sepalo','ancho_sepalo','long_petalo','ancho_petalo','clase']
df.columns = nombres

df['ancho_petalo'].plot()

df['long_petalo'].plot()

df.plot(figsize=[12,7])

df['long_petalo'].plot(kind='bar')
