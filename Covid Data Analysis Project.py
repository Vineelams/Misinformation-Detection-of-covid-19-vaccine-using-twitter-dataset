#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime


# In[3]:


covid_df =pd.read_csv("D:/covid data set/covid_19_india.csv")


# In[4]:


covid_df.head(15)


# In[5]:


covid_df.info()


# In[6]:


covid_df.describe()


# In[7]:


vaccine_df =pd.read_csv("D:/covid data set/covid_vaccine_statewise.csv")


# In[8]:


vaccine_df.head(8)


# In[12]:


covid_df.drop(["Sno", "Time","ConfirmedIndianNational", "ConfirmedForeignNational"], inplace =True, axis =1)


# In[13]:


covid_df.head()


# In[54]:


covid_df['Date'] = pd.to_datetime (covid_df['Date'], format = '%Y-%m-%d')


# In[55]:


covid_df.head()


# In[57]:


#Active cases

covid_df['Active_Cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])
covid_df.tail()


# In[31]:


covid_df.drop(["Active"], inplace =True,axis =1)


# In[34]:


covid_df.head(10)


# In[58]:


#Active cases

covid_df['Active_Cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])
covid_df.tail()


# In[43]:


print(covid_df.isnull().sum())


# In[44]:


print(covid_df.dtypes)


# In[45]:


covid_df[['Confirmed', 'Cured', 'Deaths']] = covid_df[['Confirmed', 'Cured', 'Deaths']].astype(float)


# In[46]:


covid_df['Active_Cases'] = covid_df.apply(lambda row: row['Confirmed'] - (row['Cured'] + row['Deaths']) if (row['Cured'] + row['Deaths']) < row['Confirmed'] else 0, axis=1)


# In[60]:


covid_df.tail()


# In[48]:


covid_df.dropna(how='all', inplace=True)


# In[49]:


missing_values = covid_df[['Confirmed', 'Cured', 'Deaths']].isnull().any(axis=1)
rows_with_missing = covid_df[missing_values]


# In[50]:


covid_df[['Confirmed', 'Cured', 'Deaths']] = covid_df[['Confirmed', 'Cured', 'Deaths']].fillna(0)


# In[51]:


covid_df['Active_Cases'] = covid_df['Confirmed'] - (covid_df['Cured'] + covid_df['Deaths'])


# In[59]:


print(covid_df.tail())


# In[62]:


statewise = pd.pivot_table(covid_df, values = ["Confirmed", "Deaths", "Cured"],
index = "State/UnionTerritory", aggfunc = max)


# In[63]:


statewise["Recovery Rate"] = statewise["Cured"]*100/statewise["Confirmed"]


# In[64]:


statewise["Mortality Rate"] = statewise["Deaths"]*100/statewise["Confirmed"]


# In[65]:


statewise = statewise.sort_values (by = "Confirmed", ascending = False)


# In[67]:


statewise.style.background_gradient(cmap = "magma")


# In[72]:


#Top 10 active cases states
top_10_active_cases = covid_df.groupby(by = "State/UnionTerritory").max()[['Active_Cases', 'Date']].sort_values (by = ['Active_Cases'], ascending = False).reset_index()


# In[73]:


fig = plt.figure(figsize=(16,9))


# In[75]:


plt.title("Top 10 states with most active cases in India", size = 25)


# In[80]:


ax = sns.barplot(data = top_10_active_cases.iloc[:10], y = "Active_Cases", x = "State/UnionTerritory", linewidth = 2, edgecolor = 'blue')


# In[81]:


#Top 10 active cases states
top_10_active_cases = covid_df.groupby(by = "State/UnionTerritory").max()[['Active_Cases', 'Date']].sort_values (by = ['Active_Cases'], ascending = False).reset_index()
fig = plt.figure(figsize=(16,9))
plt.title("Top 10 states with most active cases in India", size = 25)
ax = sns.barplot(data = top_10_active_cases.iloc[:10], y = "Active_Cases", x = "State/UnionTerritory", linewidth = 2, edgecolor = 'blue')
plt.xlabel("States")
plt.ylabel("Total Active Cases")
plt.show()


# In[89]:


# Top states with highest deaths
top_10_deaths = covid_df.groupby(by = 'State/UnionTerritory').max()[[ 'Deaths', 'Date']].sort_values(by =['Deaths'],ascending = False).reset_index()
fig = plt.figure(figsize=(18,5))

plt.title("Top 10 states with most Deaths", size = 25) 

ax = sns.barplot(data = top_10_deaths.iloc[:12], y = "Deaths", x="State/UnionTerritory", linewidth = 2, edgecolor = 'black')

plt.xlabel("States")
plt.ylabel("Total Death Cases")
plt.show()


# In[99]:


#Growth trend
states = ['Maharashtra', 'Karnataka', 'Kerala', 'Tamil Nadu', 'Uttar Pradesh']
filtered_df = covid_df[covid_df['State/UnionTerritory'].isin(states)]


# In[100]:


plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=filtered_df, x='Date', y='Active_Cases', hue='State/UnionTerritory')
plt.title('Active Cases Over Time for Selected States')
plt.xlabel('Date')
plt.ylabel('Active Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[102]:


vaccine_df.head()


# In[103]:


vaccine_df.rename(columns = {'Updated On': 'Vaccine_Date'}, inplace = True)


# In[104]:


vaccine_df.head(10)


# In[106]:


vaccine_df.info()


# In[118]:


vaccine_df.isnull().sum()


# In[116]:


print(vaccine_df.columns)


# In[117]:


vaccination = vaccine_df.drop(columns=['Sputnik V (Doses Administered)', 'AEFI', '18-44 Years (Doses Administered)', '45-60 Years (Doses Administered)', '60+ Years (Doses Administered)'], axis=1)


# In[120]:


vaccination.head()


# In[128]:


# Male vs Female vaccination

male = vaccination["Male(Individuals Vaccinated)"].sum()

female = vaccination["Female(Individuals Vaccinated)"].sum()

px.pie(names=["Male", "Female"], values=[male, female], title = "Male and Female Vaccination")


# In[130]:


# Remove rows where state = India

vaccine = vaccine_df[vaccine_df.State!= 'India'] 
vaccine


# In[131]:


vaccine.rename (columns = {"Total Individuals Vaccinated": "Total"}, inplace = True)
vaccine.head()


# In[136]:


# Most vaccinated State

max_vac= vaccine.groupby('State') ['Total'].sum().to_frame('Total')

max_vac= max_vac.sort_values('Total', ascending =False) [:5]

max_vac


# In[137]:


fig = plt.figure(figsize = (10,5))

plt.title("Top 5 Vaccinated States in India", size = 20)

x = sns.barplot(data = max_vac.iloc[:10], y = max_vac. Total, x = max_vac.index, linewidth=2, edgecolor='green')

plt.xlabel("States")

plt.ylabel("Vaccination")

plt.show() 


# In[ ]:




