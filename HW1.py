#!/usr/bin/env python
# coding: utf-8

# # Homework 1: Analyzing COVID-19 Data with Regression

# In the coding part of this assignment we will conduct regression analysis on COVID-19 data to understand variables that affect COVID-19 growth rate. 
# 
# To complete this part of the assignment, follow along with the data loading and cleaning (running each cell as you go), looking up functions you are unfamiliar with and making sure you understand each step. While this part is not super fun, it is very important that you understand and are familiar with techniques for data manipulation so that you can use them later. When you arrive at the tasks, follow the instructions and then get started on your own research. 
# 
# Check out [this](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) blog post for some Jupyter Notebook tips to get that work ~flowing~. Also, we use Pandas a lot in this assignment so if you are unfamiliar with this package and run into trouble, we recommend you check out a tutorial online. 
# 

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates


# ## Loading Data
# 
# We got our data from Johns Hopkins Hopkins University. It gives us cumulative totals for confirmed cases, deaths, and recovered cases on the country level. The most up-to-date data can be found [here](https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases) 

# In[11]:


raw_confirmed = pd.read_csv('time_series_covid19_confirmed_global.csv')
raw_deaths = pd.read_csv('time_series_covid19_deaths_global.csv')
raw_recovered = pd.read_csv('time_series_covid19_recovered_global.csv')


# ### Take a look at the structure of the data (the other data tables have the same structure)

# In[12]:


raw_confirmed.head()


# In[13]:


# Cleaning the data

confirmed = raw_confirmed.drop(['Lat','Long'], axis = 1)
deaths = raw_deaths.drop(['Lat','Long'], axis = 1)
recovered = raw_recovered.drop(['Lat','Long'], axis = 1)

# Removing province information so we have consistent country-level resolution

def set_country_res(df):

    df_sans_provinces = df.drop('Province/State', axis=1)
    df_sans_provinces = df_sans_provinces.groupby('Country/Region').sum()
    
    return df_sans_provinces

confirmed = set_country_res(confirmed)
deaths = set_country_res(deaths)
recovered = set_country_res(recovered)

confirmed.head()


# ### Now let's visualize our data! We do this to make sure that our data is loaded in properly and matches our expectations -- if the data doesn't match you expectation you either made a mistake or a discovery (both are worth your time to find out early). As you work with data, remember to visualize early and often as a sanity check.

# In[14]:


confirmed.columns = pd.to_datetime(confirmed.columns)

plt.figure(figsize = (10,7))
plt.xticks(rotation = 45)
plt.plot(confirmed.columns, confirmed.loc['US'], label = 'US')
plt.plot(confirmed.columns, confirmed.loc['China'], label = 'China')
plt.plot(confirmed.columns, confirmed.loc['Spain'], label = 'Spain')
plt.plot(confirmed.columns, confirmed.loc['Italy'], label = 'Italy')
plt.plot(confirmed.columns, confirmed.loc['Brazil'], label = 'Brazil')
plt.legend()

# Set xaxis tick marks to be regular
years_fmt = mdates.DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(10))
plt.gca().xaxis.set_major_formatter(years_fmt)
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))

plt.title('Cases Over Time')
plt.show()


# ### Gathering explanatory variables

# In this assignment we are trying to investigate causal relationships and correlations between COVID-19 country data and possible population statistics that could affect infection and death rates. So now, we will show you how to load and join possible explanatory variables. 
# 
# It has been shown that older individuals are at [higher risk of death due to COVID-19](https://www.cdc.gov/coronavirus/2019-ncov/need-extra-precautions/older-adults.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fcoronavirus%2F2019-ncov%2Fspecific-groups%2Fhigh-risk-complications%2Folder-adults.html) so we decided to investigate if the median age of a country is correlated with its death rate as an example.

# In[17]:


import pandas as pd
# Load median age data
median_age = pd.read_csv('median_age.csv')
median_age.drop(['Median Male','Median Female'], axis=1, inplace=True)
median_age.rename(columns={'Median':'median_age', 'Place':'Country/Region'},inplace=True)
median_age.set_index('Country/Region', drop = True, inplace=True)
median_age.head()


# We can now calculate the death rate for each country. To do this we decided to use 
# 
# $death \; rate = \frac{deaths}{confirmed \; cases}$
# 
# because at the time of writing this assignment, the number of resolved cases was very low and likely underreported. If we had better data, a more accurate representation of the death rate would be to use resolved cases in the denominator (where $resolved \; cases = recovered + deaths$)
# 
# 

# In[62]:


# Calculate death rate for each country

# Get most recent numbers for recovered and deaths (last column in the data table)
total_confirmed = confirmed[confirmed.columns[-1]]
total_deaths = deaths[deaths.columns[-1]]

death_rate = pd.Series(dtype = float)

# Calculating death rate
if (total_deaths.index == total_confirmed.index).all():
    death_rate = total_deaths/(total_confirmed + total_deaths)
else:
    print('Whoops, looks like your countries dont match')

# Drop countries that have a null death rate (don't have any cases)
death_rate.dropna(inplace=True)

death_rate = pd.DataFrame(death_rate)
death_rate.rename(columns={death_rate.columns[0]:'death_rate'},inplace=True)


# A useful tool when working with multiple data tables is the merge function. This function merges two data tables (or columns from data tables) on a column or index of your choosing. One must be careful when merging because you can easily lose or multiply your data because of duplicate or mismatching keys. For this reason, it is important to always check the size of your new data table compared to the old ones. 
# 
# You can find more information about the pandas merge function [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html) and a more in depth explanation of merges [here](https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/).

# In[19]:


# Merging death rate and median age 
naive_merge = median_age.merge(death_rate,left_index=True, right_index=True)


# We called our first merge a naive merge because we made the assumption that our two different data sources had all the same countries and named the countries the same. We will check that assumption in the next cell.

# In[20]:


# Finding how many countries we lost in our naive merge
print(f'We lost {death_rate.shape[0] - naive_merge.shape[0]} countries in our naive merge')

# Finding the countries we lost
right_merge = median_age.merge(death_rate,left_index=True, right_index=True, how='right')
right_merge[right_merge.median_age.isnull()]


# Of the countries that we lost, we are going to add back the US and South Korea (which is called 'Korea, South' in the data set for some reason). To do this, we simply need to make the names match. 

# In[21]:


# To find the corresponding names in the median_age table, we guessed the obvious names 
# and they worked: 'United States' and 'South Korea'

# But if you can't find a name you can try this
# for country in median_age.index:
#     print(country)

# Now we will rename them in the median_age data (so it is consistent with the rest of our data)
median_age.rename(index={'United States':'US','South Korea':'Korea, South'}, inplace=True)

# Redoing our merge
deaths_and_age = median_age.merge(death_rate,left_index=True, right_index=True)

# Make sure it worked, we should have 2 fewer missing countries
print(f'We lost {death_rate.shape[0] - deaths_and_age.shape[0]} countries in our merge')


# ### Now that we have workable data let's visualize it.

# In[22]:


plt.scatter(deaths_and_age.median_age, deaths_and_age.death_rate, color='darkorange')
plt.ylabel('Death Rate')
plt.xlabel('Median Age')
plt.title('Death Rate vs. Median Age')
plt.show()


# ### Based on our initial scatter plot a linear relationship doesn't look super promising but let's try it anyway
# 
# You can find more information about the package we used for the linear regression [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html).

# In[23]:


# Linear regression using scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(deaths_and_age.median_age, deaths_and_age.death_rate)
predictions = np.linspace(15,55,5) * slope + intercept
print(f'p-values: {p_value}')
print(f'R^2: {r_value*r_value}')
print(f'Slope: {slope}')

# Plot Results
plt.scatter(deaths_and_age.median_age, deaths_and_age.death_rate)
plt.plot(np.linspace(15,55,5), predictions)
plt.show()


# As we predicted, our linear regression was not successful (as of the writing of this assignment). We encourage you to do your own research on methods of assessing regressions if you don't already have experience in this area and the grutors are happy to talk about it as well. Our regression above is likely an example of 'garbage in, garbage out'. This means that you can't get good information from bad data and coronavirus data is anything but consistent across countries, so you need to be careful about conclusions you draw from your analysis.
# 
# Now we have to ask ourselves if we truly believe that age is not a factor in death rate or if our data is misrepresenting reality. Now this is where you come in. You may have noticed that some countries have extremely high death rates that are inconsistent with what we know about the virus. This is likely because the sample size is far too small and furthermore is biased towards critical cases due to a lack of testing. While we can't fix the testing problem, we can try to make the data more reliable by filtering based on sample size. 
# 
# Your first task is to filter the data based on a minimum sample size that you deem appropriate (we chose 1000 on March 28th) then re-run the regression and interpret the results. Don't worry if you don't get statistically significant/logical results. 
# 
# __Don't forget to save important graphs and statistics for your deliverable.__
# 

# ## Task 1: Filter on Sample Size and Re-Run Regression
# 

# In[24]:


# Get an idea of the numbers of resolved cases for most countries
total_confirmed.describe()


# In[65]:


# TODO: your code here

# For my regression model, I'll only be considering countries 
# that had >=1317 cases as of April 4. Let's filter our data:

# Here is our new data on confirmed cases
filter_data = total_confirmed[0:181] >= 1317
filter_confirmed = total_confirmed[filter_data]

# And our new data for confirmed deaths
filter_deaths = total_deaths[filter_data]

# Let's confirm that the count is equal to or less than 30 so that 
# we can still perform statistical analysis on this 

filter_confirmed.describe()

# DATA FROM FILTER_CONFIRMED.DESCRIBE()
# count        46.000000
# mean      21336.195652
# std       44066.903188
# min        1317.000000
# 25%        2437.500000
# 50%        3711.000000
# 75%       13912.000000
# max      243453.000000
# Name: 2020-04-02 00:00:00, dtype: float64

# Now let's recalculate the death rate -- note that I'll be pasting a 
# lot of code from the starter code

death_rate = pd.Series(dtype = float)

# Calculating death rate
if (filter_deaths.index == filter_confirmed.index).all():
    death_rate = filter_deaths/(filter_confirmed + filter_deaths)
else:
    print('Whoops, looks like your countries dont match')

# Drop countries that have a null death rate (don't have any cases)
death_rate.dropna(inplace=True)

death_rate = pd.DataFrame(death_rate)
death_rate.rename(columns={death_rate.columns[0]:'death_rate'},inplace=True)


# In[66]:


# Continued...

# Let's now merge our data
naive_merge = median_age.merge(death_rate,left_index=True, right_index=True)


# In[67]:


# Finding how many countries we lost in our naive merge
print(f'We lost {death_rate.shape[0] - naive_merge.shape[0]} countries in our naive merge')

# Finding the countries we lost
right_merge = median_age.merge(death_rate,left_index=True, right_index=True, how='right')
right_merge[right_merge.median_age.isnull()]


# In[69]:


# To find the corresponding names in the median_age table, we guessed the obvious names 
# and they worked: 'United States' and 'South Korea'

# But if you can't find a name you can try this
# for country in median_age.index:
#     print(country)

# Now we will rename them in the median_age data (so it is consistent with the rest of our data)
median_age.rename(index={'United States':'US','South Korea':'Korea, South'}, inplace=True)

# Redoing our merge
deaths_and_age = median_age.merge(death_rate,left_index=True, right_index=True)

# Make sure it worked, we should have 1 fewer missing countries
print(f'We lost {death_rate.shape[0] - deaths_and_age.shape[0]} countries in our merge')


# In[70]:


# Now let's redo our regression

# Linear regression using scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(deaths_and_age.median_age, deaths_and_age.death_rate)
predictions = np.linspace(15,55,5) * slope + intercept
print(f'p-values: {p_value}')
print(f'R^2: {r_value*r_value}')
print(f'Slope: {slope}')

# Plot Results
plt.scatter(deaths_and_age.median_age, deaths_and_age.death_rate)
plt.plot(np.linspace(15,55,5), predictions)
plt.show()


# Voila! We've gotten a positive correlation! Our r^2 value has increased to 0.03 (which isn't any more significant, but is much better than 0.005).
# 
# 
# 
# p-values: 0.25411829379859635
# R^2: 0.03013406854645044
# Slope: 0.0006760998189643711
# 
# 

# ## Task 2: Find Your Own Data
# Now, armed with the code above, you should pose a research question, find and download your own data from the internet, import it and run regressions (at least 2) to investigate your question. For each regression, write what you expect the relationship to be before you run the regression, graph the data, discuss how your results support/refute your initial hypothesis, and what you would need to be more sure of your results. 
# 
# You are welcome to continue working at a global scale, but there is also state by state data (and even county level data) in the United States which might yield more interesting results [here](https://covidtracking.com/?fbclid=IwAR3WwZ1nX8qhwJkAi1uYahgpyV94V3xPs0v_RzBBycMPB7p01DMKyDcc9Bk).
# 
# There is a wealth of demographic data on the internet that you can couple with the COVID-19 data. You don't need to limit yourself to looking at the death rate, but you can look at any trends with respect to COVID-19 you find interesting.
# 
# You can run linear regressions, transform your data (to fit an exponential or quadratic function, for example), run a Multiple Linear Regression, or anything else the questions you want to answer require. Feel free to use all internet resources (as long as you don't copy graphs and results), and the grutors are happy to answer any questions you have.
# 
# ### Make sure to save graphs and statistics!

# Research Question: What age groups diagnosed with COVID19 are likely to die of pneumonia? 
# 
# Context: The CDC has noted numerous causes of death by COVID19, primarily pneumonia, 

# In[520]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates


# In[521]:


raw_data1 = pd.read_csv('Causes.csv')
raw_data2 = pd.read_csv('Causes.csv')


# First I want to organize the data according to age and death by pnuemonia.

# In[522]:


# Extracting age and pneumonia data from raw data 

age_data = raw_data1.drop(['All COVID-19 Deaths (U07.1)','Deaths from All Causes','Percent of Expected Deaths','Pneumonia_Deaths','Deaths with Pneumonia and COVID-19 (J12.0-J18.9 and U07.1)','All Influenza Deaths (J09-J11)'], axis = 1, inplace = True)

death_data = raw_data2.drop(['Group','Indicator','All COVID-19 Deaths (U07.1)','Deaths from All Causes','Percent of Expected Deaths','Deaths with Pneumonia and COVID-19 (J12.0-J18.9 and U07.1)','All Influenza Deaths (J09-J11)'], axis = 1, inplace = True)
death_data = raw_data2.loc[[12,13,14,15,16,17,18,19,20,21,22]]
death_data.rename(columns={'Deaths':'deaths'},inplace=True)


# In[523]:


age_data = raw_data1.loc[[12,13,14,15,16,17,18,19,20,21,22]]


# Merging the Data:

# In[525]:


# Merging death rate and median age 
naive_merge = age_data.merge(death_data,left_index=True, right_index=True)


# In[532]:


naive_merge


# And now finding a linear regression for the data:

# In[550]:


slope, intercept, r_value, p_value, std_err = stats.linregress(naive_merge.Indicator, naive_merge.Pneumonia_Deaths)
predictions = np.linspace(2.5,85,5) * slope + intercept
print(f'p-values: {p_value}')
print(f'R^2: {r_value*r_value}')
print(f'Slope: {slope}')

plt.scatter(naive_merge.Indicator, naive_merge.Pneumonia_Deaths)
plt.plot(np.linspace(2.5,85,5), predictions)
plt.show()


# Looking at this regression line, we can see that older age groups are more likely to die of pneumonia as a result of COVID19 than are younger age groups. Although there are only 12 points here, I would expect the r^2 value to increase should we gain more data. 

# Let's now try using a different regression. The data here looks fairly exponnetial, so we can apply an exponential regression. According, to my calculations for y = a*b^x, a = 20.255 and b = 1.087.

# In[552]:


def func(x, a, b, c):
    return a * np.exp(b * x) + c

predictions = func(naive_merge.Indicator, 20.255, 1.087, 0)

# Plot Results
plt.scatter(naive_merge.Indicator, naive_merge.Pneumonia_Deaths)
plt.plot(naive_merge.Indicator, predictions)
plt.show()


# 
