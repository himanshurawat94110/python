#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:00:15 2020

@author: himanshurawat
"""
## Q1
import numpy as np
import pandas as pd 
pd.show_versions()
pd.__version__

###### Q2 - convert index of series to a column of a dataframe 

import numpy as np
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
series = pd.Series(mylist)
series =pd.Series(myarr)
series=pd.Series(mydict)
pd.Series(mydict)
pd.DataFrame(series)


#### Q3 

mydict = dict(zip(mylist, myarr))
series=pd.Series(mydict)

##sol1
dataframe = pd.DataFrame(series,index = series.index)
dataframe['index']=dataframe.index


#sol2
dataframe.reset_index(level = 0 ,inplace =True)

####sol3
df = series.to_frame().reset_index()




##### Q4
#create dataframe frm two series 
import numpy as np
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))

dataframe =pd.DataFrame([ser1,ser2])
dataframe_1 =pd.concat([ser1,ser2],axis=1)

########  Q5
##rename series 
ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'),name = 'alphabets')
ser.rename('alphabets')
##or 

ser.name('alphabets')  ## if not names till then 


#######  Q6 and Q7 

####setdiff function for python 

ser1= pd.Series([1,2,3,4,5])
ser2=pd.Series([5,6,7,8,9])
import numpy as np

# sol 1 and 2 for Q6
main_list =list(set(ser1)-set(ser2))   
ser1[~ser1.isin(ser2)]
main_series = pd.Series([])


## sol for both Q6 and Q7 

ser3=pd.Series([])
ser4=pd.Series([])

for i in ser1.values:
    if i not in ser2.values:
        ser3=pd.Series([i])
        ser4=ser4.append(ser3)

print(ser4)

for i in ser2.values:
    if i not in ser1.values:
        ser3=pd.Series([i])
        ser4=ser4.append(ser3)
        
print(ser4)

# sol for Q7 

ser_u = pd.Series(np.union1d(ser1, ser2))  # union
ser_i = pd.Series(np.intersect1d(ser1, ser2))  # intersect
ser_u[~ser_u.isin(ser_i)]



###### Q8

ser = pd.Series(np.random.normal(10, 5, 25))     
ser.quantile([0.00,0.25,0.5,0.75,1.00])
ser.max()
ser.min()


####     Q9

ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
pd.value_counts(ser)


np.random.RandomState(100)
ser=pd.Series(np.random.randint(1,5,size = 12))
ser
###find max and second max


########    Q10


unique_ser=pd.Series(ser.value_counts())
unique_ser
unique_ser.sort_values( ascending =False)


#second_max=0

#for n in unique_ser.values:
    
 #   if n>second_max and n<unique_ser.max():
  #      second_max=n
#second_max
#new series 

#ser_2 =pd.Series([unique_ser.max(),second_max])
#ser_2

glass=pd.Series(unique_ser[2::].index)
glass


for i in glass.values:
    if i in ser.values:
        ser.replace(to_replace =i,value='other',inplace=True)
    
#ser.sort_values(ascending = True).unique()
#ser.unique()




###### Q11
ser = pd.Series(np.random.random(20))

ram = pd.DataFrame(ser)

ram =   pd.qcut(ser,q=10)


###  Q12


ser = pd.Series(np.random.randint(1, 10, 35))

data = pd.DataFrame(ser).values.reshape(7,5)


###### Q13

ser = pd.Series(np.random.randint(1, 10, 7))

x = pd.Series(ser.index)

for i in x:    
    if ser[[i]]%3 == 0:
    print(i)
        
    
####  Q14
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]

war = ser.iloc[pos]

##### Q15
ser1 = pd.Series(range(5))
ser2 = pd.Series(list('abcde'))

ser_hori=pd.concat([ser1,ser2],axis =1)
ser_ver=pd.concat([ser1,ser2],axis = 0)


ser_new=pd.DataFrame(ser1)
ser_new['new']=ser2.values.tolist()



####### Q16

ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])


for i in ser2.values:
    if i in ser1.values:
        print(ser1[ser1==i].index[0])


######  Q17      
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)
war = ((truth - pred)**2)/10 


##########  Q18     

ser = pd.Series(['how', 'to', 'kick', 'ass?'])


ser.str.capitalize()


for i in ser.valuues :
    return(i.title())

ser.map(lambda x: x.title())
ser.map(lambda x: x[0].upper()  + x[1:] )

pd.Series([i.title() for i in ser])


########     Q19 
ser = pd.Series(['how', 'to', 'kick', 'ass?'])

ser.map(lambda x: len(x))




#############  Q20 

ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])

ser.index
war = pd.Series

for i in ser.index :
    print(i)

def myfunc(x,y):
    x.iloc[y+1]-x.iloc[y]


war = ser.diff(periods=1)
war_2 = war.diff(periods=1)
ser.diff(periods=2)



#######   Q21  ##########


from datetime import datetime
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])

ser.map(lambda x: pd.Timestamp(x))

from dateutil.parser import parse
ser_ts =ser.map(lambda x: parse(x))
ser_ts


#######   Q22  #####  
ser_ts.dt.day
ser_ts.dt.week
ser_ts.dt.dayofyear
dow = ser_ts.dt.dayofweek
peace = pd.DataFrame(dow, columns=['Day numbr'])

data = [[1,'Monday'],[2, 'Tuesday'], [3,'wednesday'],[4,'Thursday'],[5,'Friday'],[6,'Saturday'],[7,'Sunday']]

war = pd.DataFrame(data,columns=['Day number','Day'])


peace['Dayy']=peace['Day numbr'].map(war['Day'])

######  Q23 

ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])


## my sol 
from dateutil.parser import parse
ser_ts =ser.map(lambda x: parse(x))
ser_ts
new = ser_ts.apply(lambda x: x.replace(day = 4))


## given solutions 

# Solution 1
from dateutil.parser import parse
# Parse the date
ser_ts = ser.map(lambda x: parse(x))

# Construct date string with date as 4
ser_datestr = ser_ts.dt.year.astype('str') + '-' + ser_ts.dt.month.astype('str') + '-' + '04'

# Format it.
[parse(i).strftime('%Y-%m-%d') for i in ser_datestr]

# Solution 2
ser.map(lambda x: parse('04 ' + x))


#### Q24 

####  importing regex module into pyhton which is defined as re ## 
import re

ser = pd.Series(['Aapple', 'Orange', 'Pln', 'Python', 'Money'])


### what i Did, solution 
vowels = pd.Series(['a','e','i','o','u'])
regexpattern = re.compile('[aeiou]')

listofmatches_2=[]

for x in ser:
    listofmatches =regexpattern.findall(x)
    listofmatches_2.append(listofmatches)
    print(listofmatches)

# pandas 101 solution to the same 
    
from collections import Counter
mask = ser.map(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
ser[mask]

#war = np.zeros(shape = (5,5))
#war_1 =pd.DataFrame(war,index =[vowels])
# breakup of above almbda function peicewise 
# also suum is defined as integer here outside the for loop but inside the for loop during
# summation it will convert to a seires because for append in series can only happen if
#it itself is a series and if we do we get the following error 
#TypeError: cannot concatenate object of type '<class 'int'>'; only Series and DataFrame objs are valid
 
suum_1=pd.Series()
suum=0
count=0
for x in ser.index:
    for i in list('aeiou'):
        count = sum([pd.Series([Counter(ser[x].lower()).get(i,0)])])
        suum=sum([count,suum])
    suum_1=suum_1.append(suum)

### 
                #war_1[x:x+1]=count_1
   # count_1=pd.Series()
    
        
for x in ser.index:
    print(x)
    
###### Q25 -- filter valid emails from a series of emails 

emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'

#my solution 
pattern_1= re.compile(pattern)

listofmatches_2=[]
for x in emails:
    listofmatches = pattern_1.findall(x)
    listofmatches_2.append(listofmatches)
    print(listofmatches)

#given solution 
import re
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
mask = emails.map(lambda x: bool(re.match(pattern, x)))
emails[mask]

# Solution 2 (as series of list)
emails.str.findall(pattern, flags=re.IGNORECASE)

# Solution 3 (as list)
[x[0] for x in [re.findall(pattern, email) for email in emails] if len(x) > 0]

#Q26

fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())
print(fruit.tolist())

#creating a dataframe 
data = pd.concat([fruit,weights],axis = 1)
#renaming th columns 
data.rename(columns = {0:'fruit',1:'weigths'},inplace = True)

#using groupby 
data_1=data.groupby('fruit')['weigths'].count()
data_2 =data.groupby('fruit')['weigths'].mean()


#######Q27

from scipy.spatial import distance
p1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
p2 = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

d = distance.euclidean(p1,p2)

## solution 2 

import math
distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

# use of zip 

names = ['Bob','Jessica','Mary','John','Mel']
births = [968, 155, 77, 578, 973]
BabyDataSet = list(zip(names,births))

#defining a dataframe and assigning columns 

data =pd.DataFrame(BabyDataSet,columns = ['Name','births']) 

#finding column names 

for col in data.columns :
    print(col)


# Q28 - find a local maxima for the series 
    
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
# printingn solution and multiple values at a time 
for i in range(1,max(ser.index)):
    if ser[i]>ser[i-1] and ser[i]>ser[i+1]:
        print(ser[i],ser.index[i])


dd = np.diff(np.sign(np.diff(ser)))
peak_locs = np.where(dd == -2)[0] + 1
peak_locs


# Q 29 
        
my_str = 'dbc deb abed gade'
list_1=list(my_str)
series_1=pd.Series(list_1)

# how to checking the frquency 
series_1.value_counts()
#  how to checking the unique values '

#my solution 
for i in series_1.index:
    if series_1[i] ==' ':
        series_1[i]='c'

# 101 solution
series_1.replace(' ', 'c')


series_1.unique()

# How to checking frequency across a dataframe 

fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
#print(weights.tolist())
#print(fruit.tolist())

#creating a dataframe 
data = pd.concat([fruit,weights],axis = 1)
#renaming th columns 
data.rename(columns = {0:'fruit',1:'weigths'},inplace = True)

#checking a frequency of item  (both method works)
data['fruit'].count()

data.groupby('fruit').count()

data['fruit'].value_counts()
# checkign the mean of the column after groupwse 
# since no column is defined so it gives mean of all columns 

data.groupby('fruit').mean()

# applying frequenct function to all columns 
data.apply(pd.value_counts)

## Q30 

s = pd.Series(np.random.randint(1,100,10),pd.date_range('2001-01-01',periods=10, freq='W-mon').to_series())


#Q31 - find na value and replace with previous row value 
ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))

ser_1 = pd.Series(np.where(ser.isna()))

for i in ser_1:
    ser[i]=ser[i-1]
    
# 101 solution 
    
    # Solution
ser.resample('D').ffill()  # fill with previous value

# Alternatives
ser.resample('D').bfill()  # fill with next value
ser.resample('D').bfill().ffill()  # fill next else prev value

#Q32
ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))
ser.autocorr(lag =2)
ser_2 =[]
for i in range(1,10):
    ser_1=pd.Series(ser.autocorr(lag=i))
    ser_2.append(ser_1)
    print(ser.autocorr(lag=i))

ser_2

#101 solution 
autocorrelations = [ser.autocorr(i).round(2) for i in range(11)]
print(autocorrelations[1:])
print('Lag having highest correlation: ', np.argmax(np.abs(autocorrelations[1:]))+1)



ser.autocorr(lag =9 )
#Q######## 33
######## my solution 
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', skiprows =lambd x: logic(x))

def logic(index):
    if index % 50 ==0:
        return False 
    return True 
    
######### solutions from the pandas 101 
    #solution 1
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
df2 = pd.DataFrame()
for chunk in df:
    df2 = df2.append(chunk.iloc[0,:])


# Solution 2: Use chunks and list comprehension
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
df2 = pd.concat([chunk.iloc[0] for chunk in df], axis=1)
df2 = df2.transpose()

# Solution 3: Use csv reader
import csv          
with open('BostonHousing.csv', 'r') as f:
    reader = csv.reader(f)
    out = []
    for i, row in enumerate(reader):
        if i%50 == 0:
            out.append(row)

df2 = pd.DataFrame(out[1:], columns=out[0])
print(df2.head())

for col in df2.columns:
    print(col)

######### 34 
## my solution 
    df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
for col in df.columns:
    print(col)
    
for i in df['medv'].index:
    if df['medv'].iloc[i]>25:
        df['medv'].iloc[i]='High'
    else:
        df['medv'].iloc[i]='Low'
### 101 solution 
## solution 1  
      df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', 
                 converters={'medv': lambda x: 'High' if float(x) > 25 else 'Low'})


# Solution 2: Using csv reader  ### same as question 33 solution 
import csv
with open('BostonHousing.csv', 'r') as f:
    reader = csv.reader(f)
    out = []
    for i, row in enumerate(reader):
        if i > 0:
            row[13] = 'High' if float(row[13]) > 25 else 'Low'
        out.append(row)

df = pd.DataFrame(out[1:], columns=out[0])
print(df.head())

###### Q35 
### rows as strides 
L = pd.Series(range(15))

##### OKKKK

def gen_str(a, gap = 3 ,  length = 5):
    n_strides = ((a.size-length)//gap)+1
    return np.array([a[s:(s+length)] for s in np.arange(0,a.size,gap)[:n_strides]])

gen_str(L,gap=4,length=6)


def gen_strides(a, stride_len=5, window_len=5):
    n_strides = ((a.size-window_len)//stride_len) + 1
    return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])

gen_strides(L, stride_len=2, window_len=4)


np.arange(0, a.size, stride_len)
np.arange(0, a.size, stride_len)[:n_strides]
n_strides = ((a.size-window_len)//stride_len) + 1

## Q 36 

#### my Solution 1
  df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',usecols = ['medv', 'crim'])    

### my sol 2 
  ##### use usecols calable function 
y =('medv','crim')
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv',usecols =lambda column: column in y )    


##### Q37 
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

## number of rows and columns 
df.shape
## summary statistics, for complete database together.
type(df)
### for one specific column use dtype, for all culumns at once use dtypes 
df.dtypes
df['Model'].dtype

#how many columns under each type 
print(df.get_dtype_counts())
print(df.dtypes.value_counts())

# summary statistics
df_stats = df.describe()
#
# numpy array 
df_arr = df.values

# list
df_list = df.values.tolist()


#### Q38 
#### my solution 
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

dll=df['Price'].sort_values(ascending =False)
df[['Model','Price']].iloc[dll.index[0]]

## only gives max value and not inex positiosn 
index = df['Price'].idxmax(skipna=True)
value = df['Price'].max()

df[['Model','Type']].iloc[index]


#### pandas 101 solutions 


# Solution
# Get Manufacturer with highest price
df.loc[df.Price == np.max(df.Price), ['Manufacturer', 'Model', 'Type']]

# Get Row and Column number
row, col = np.where(df.values == np.max(df.Price))

# Get the value
df.iat[row[0], col[0]]
df.iloc[row[0], col[0]]

# Alternates
df.at[row[0], 'Price']
df.get_value(row[0], 'Price')

# The difference between `iat` - `iloc` vs `at` - `loc` is:
# `iat` snd `iloc` accepts row and column numbers. 
# Whereas `at` and `loc` accepts index and column names.

########  Q 39 ### rename a specific column in a dataframe  

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df
df.rename(columns={'Model':'Modelling'})  

### Pandas column rename for all coluns function 

df.columns = df.columns.map(lambda x: x.replace('.', '_'))
print(df.columns)

##### how to get columns 

df.columns 

##### Q 40 

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df.isnull()
df.isnull().any()
df.isnull().sum()
df.isnull().values.any()
df.isnull().sum().sum()


## Q 41 which column has the maximum number of NAs 
#####
n_missings_each_col = df.apply(lambda x: x.isnull().sum())
n_missings_each_col.argmax()


## my solution 
rr =df.isnull().sum()
rr.sort_values(ascending= True)

### Q 42 replace multiple values with mean
### my solution from this place 
#isnull(): Generate a boolean mask indicating missing values
#notnull(): Opposite of isnull()
#dropna(): Return a filtered version of the data
#fillna(): Return a copy of the data with missing values filled or imputed



df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

df.isnull()
df2 = df.fillna(0)
df2.isnull().any()
df2.isnull().sum()
df2.isnull().sum().sum()
## my solution from here 
https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html
Method              Action
pad / ffill         Fill values forward
bfill / backfill    Fill values backward
With time series data, using pad/ffill is extremely common so that the “last known value” is available at every time point.

ffill() is equivalent to fillna(method='ffill') and bfill() is equivalent to fillna(method='bfill')

df.fillna(0)
df.fillna('missing')

#######replace with integer 0 or string 'missing' depending on whetehr datatype is float64 or object 

for col in df.columns:
    print(col)
    if type(df[col])

df['Min.Price'].fillna(method ='ffill')

df.fillna(df.mean())

df22=df.fillna(df.mean()['Min.Price':'Max.Price'])
df22.isnull()


## pandas 101 solution 

# Input
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# Solution
df_out = df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']].apply(lambda x: x.fillna(x.mean()))
print(df_out.head())


### Q43

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

#### my solution 
df['Min.Price'].fillna(df['Min.Price'].mean())
df['Max.Price'].fillna(df['Max.Price'].median())


#### my solution alternative
### apparently apply function does not work with single square bracket
df2=df[['Min.Price']].apply(lambda x: x.fillna(x.mean()))
df3=df[['Max.Price']].apply(lambda x: x.fillna(x.median()))


###### pandas 101 solution 

# Input
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

d = {'Min.Price': np.nanmean, 'Max.Price': np.nanmedian}
df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']].apply(lambda x, d: x.fillna(d[x.name](x)), args=(d, ))



### Q44 filter a column from a dataframe as a dataframe and not as series 

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde')) 

# my solution 
df2=pd.DataFrame(df['a'])


# 101 solution

# Solution
type(df[['a']])
type(df.loc[:, ['a']])
type(df.iloc[:, [0]])

# Alternately the following returns a Series
type(df.a)
type(df['a'])
type(df.loc[:, 'a'])
type(df.iloc[:, 1])
 
### Q45

# Input
df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

# Solution Q1
df[list('cbade')]

# Solution Q2 - No hard coding
def switch_columns(df, col1=None, col2=None):
    colnames = df.columns.tolist()
    i1, i2 = colnames.index(col1), colnames.index(col2)
    colnames[i2], colnames[i1] = colnames[i1], colnames[i2]
    return df[colnames]

df1 = switch_columns(df, 'a', 'c')

# Solution Q3
df[sorted(df.columns)]
# or
df.sort_index(axis=1, ascending=False, inplace=True)

##### Q46 

# Input
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# Solution
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
# df

# Show all available options
 pd.describe_option()    


# 47 

# Input
 
df = pd.DataFrame(np.random.random(4)**10, columns=['random'])

# Solution 1: Rounding
df.round(4)

# Solution 2: Use apply to change format
df.apply(lambda x: '%.4f' % x, axis=1)
# or
df.applymap(lambda x: '%.4f' % x)

# Solution 3: Use set_option
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Solution 4: Assign display.float_format
pd.options.display.float_format = '{:.4f}'.format
print(df)

# Reset/undo float formatting
pd.options.display.float_format = None


#### Q 48 

