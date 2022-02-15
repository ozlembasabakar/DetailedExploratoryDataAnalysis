#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install statsmodels


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import pylab
from scipy.stats import skew, kurtosis


# In[3]:


df = pd.read_csv('train.csv')
df.head()


# In[3]:


df.info()


# In[1]:


df.isnull().sum()


# 'Id' and columns with a NULL data count of less than 0.3 should be removed.

# In[4]:


df2 = df[[column for column in df if df[column].isnull().sum() / len(df) < 0.3]]
del df2['Id']
print("Dropped columns:")
for c in df.columns:
    if c not in df2.columns:
        print(c, end=", ")


# In[53]:


df2.info()


# # Normal Distribution

# We only need numeric variables to test the normal distribution.

# In[5]:


# df_num = df2.select_dtypes(include = ['float64', 'int64'])
df_num = df2.select_dtypes(include = ['number'])
df_num.head()


# ## Graphical Methods:

# In[18]:


print(df['SalePrice'].describe())
plt.figure(figsize=(16, 20))
sns.distplot(df['SalePrice'], color='b', bins=50).set_title('SalePrice')


# In[19]:


df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
plt.suptitle("Histogram Plots for all Numeric Variables")


#   

# The distributions of the 'LotFrontage', '1stFlrSF', 'TotalBsmtSF', 'GarageArea' variables are similar to the SalePrice variable. Let's look at these variables in more detail.

#   

# In[41]:


def variable_vis(df, var_list):
    for i in var_list:
        plt.figure(figsize=(16, 20))
        sns.distplot(df[str(i)], color='b', bins=50).set_title(str(i))


# In[42]:


var_list = ('LotFrontage', '1stFlrSF', 'TotalBsmtSF', 'GarageArea')
variable_vis(df_num, var_list)


#   

# ## Statistical Tests

# For normal distribution the skewness should be 0 and the kurtosis should be 3. 
# 
# For Pearson’s definition, the kurtosis value for normal distribution is 3. That's why we should add  'fisher=False'  to the code. 

# In[43]:


for i in df_num.columns:
    print(i,'skewness is: ',skew(df_num[[i]]))
    print(i,'kurtosis is: ',kurtosis(df_num[[i]], fisher=False))
    print('------------------------------------------------------')


#   

# #### Let's look at every variable skewness and kurtosis results: 
# 
# - MSSubClass has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - LotArea has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - OverallQual has no tail, flatness, sharpness. That means Symmetrical skewnessand and Mesakurtic(Normal) distribution.
# 
# - OverallCond has no tail and has a sharp peak. That means Symmetrical skewnessand and Leptokurtic distribution.
# 
# - YearBuilt has no tail, flatness, sharpness. That means Symmetrical skewnessand and Mesakurtic(Normal) distribution.
# 
# - YearRemodAdd has no tail and has a quite flatness. That means Symmetrical skewnessand and Platykurtic distribution.
# 
# - BsmtFinSF1 has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - BsmtFinSF2 has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - TotalBsmtSF has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - 1stFlrSF has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - 2ndFlrSF has no tail, flatness, sharpness. That means Symmetrical skewnessand and Mesakurtic(Normal) distribution.
# 
# - LowQualFinSF has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - GrLivArea has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - BsmtFullBath has no tail and has a quite flat. That means Symmetrical skewnessand and Platykurtic distribution.
# 
# - BsmtHalfBath has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - FullBath has no tail and has a quite flat. That means Symmetrical skewnessand and Platykurtic distribution.
# 
# - HalfBath has no tail and has a quite flat. That means Symmetrical skewnessand and Platykurtic distribution.
# 
# - BedroomAbvGr has no tail and has a quite sharp peak. That means Symmetrical skewnessand and Leptokurtic distribution.
# 
# - KitchenAbvGr has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - TotRmsAbvGrd has a thick left tail and a quite sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - Fireplaces has no tail, flatness, sharpness. That means Symmetrical skewnessand and Mesakurtic(Normal) distribution.
# 
# - GarageCars has no tail, flatness, sharpness. That means Symmetrical skewnessand and Mesakurtic(Normal) distribution.
# 
# - GarageArea has no tail and has a quite sharp peak. That means Symmetrical skewnessand and Leptokurtic distribution.
# 
# - WoodDeckSF has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - OpenPorchSF has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - EnclosedPorch has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - 3SsnPorch has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - ScreenPorch has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - PoolArea has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - MiscVal has a thick left tail and a sharp peak. That means Negative skewness and Leptokurtic distribution.
# 
# - MoSold has no tail, flatness, sharpness. That means Symmetrical skewnessand and Mesakurtic(Normal) distribution.
# 
# - YrSold has no tail and has a quite flat. That means Symmetrical skewnessand and Platykurtic distribution.

# ![Relationship_between_mean_and_median_under_different_skewness.png](attachment:Relationship_between_mean_and_median_under_different_skewness.png)

# ![1_lU3NEdbwWeGoJyuYfmuQVw.png](attachment:1_lU3NEdbwWeGoJyuYfmuQVw.png)

# Positive skewness: Thicker right tail and mode < median < mean.
# 
# Negative skewness: Thicker left tail and mean < median < mode.
# 
# Symmetrical skewness: mode = median = mean.
# 
# 
# ##### Excess Kurtosis for Normal Distribution = 3 – 3 = 0
# Types of Excess Kurtosis:
# 
#     1. Positive excess kurtosis: The distribution has a sharp peak and is called a Leptokurtic distribution.
#     
#     2. Negative excess kurtosis: The distribution has quite flat and is called a Platykurtic distribution.
# 
#     3. The distribution is neither flat nor sharp. and is called the Mesakurtic(Normal) distribution.

# In[44]:


alpha = 0.05


# #### Jarque-Bera
# This test looks for skewness = 0 and kurtosis = 3 (normal distribution values). The test statistic is not always negative. It shows that data does not have a normal distribution if it is far from zero.
# 
# H0: The sample comes from a normal distribution. S=0 and K=3.
# 
# H1: The sample is not coming from normal distribution. 

# In[45]:


for i in df_num.columns:
    print ([i])
    a,b= stats.jarque_bera(df[[i]])
    print ("Statistics", a, "p-value", b)
    if b < alpha:  
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")


# #### Kolmogorov-Smirnov
# This is a non-parametric test. That means, it has no assumption about the distribution of the data. It is tested if samples X1, X2,..., Xn and Y1, Y2,..., Yn come from the same distribution.
# 
# H0: Fx(z) is equal to Fy(z)
# 
# H1: Fx(z) is not equal to Fy(z)

# In[46]:


for i in df_num.columns:
    print ([i])
    a,b= stats.kstest(df_num[[i]], 'norm')
    print ("Statistics", a, "p-value", b)
    if b < alpha:  
        print("The null hypothesis can be rejected")
    else:
        print("The null hypothesis cannot be rejected")


# The result of the Jarqu-Bera test statistic and Kolmogorov-Smirnov test statistic are, for LotFrontage, MasVnrArea, GarageYrBlt variables the null hypothesis cannot be rejected. But for all other variables can be rejected. The fact that the test statistics and p-values were NAN could explain why the null hypothesis can not be rejected for LotFrontage, MasVnrArea, GarageYrBlt variables.

# Early results can be accessed using graph approaches for testing normality. The statistical results of the tests provide scientific information.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Heteroskedasticity

# In[ ]:





# # Autocorrelation python

# In[ ]:




