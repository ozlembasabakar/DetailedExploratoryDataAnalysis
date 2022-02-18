#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install statsmodels


# In[44]:


pip install plotly


# In[150]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import pylab
from scipy.stats import skew, kurtosis
import operator
import plotly.express as px
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats


# In[2]:


df = pd.read_csv('train.csv')
df.head()


# In[3]:


df.info()


# In[1]:


df.isnull().sum()


# 'Id' and columns with a NULL data count of less than 0.3 should be removed.

# In[3]:


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

# In[4]:


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

# # Correlation

# In[19]:


df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice
df_num_corr


# In[73]:


golden_features = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features), golden_features))


# We have 10 strongly correlated features with SalePrice. But we should look at the outliers. 
# 
# Correlation isn't always enough to explain the relationship between data. 
# Plotting would show which features have outliers. Also, without these outliers, we should check the correlation.

# In[24]:


for i in range(0, len(df_num.columns), 6):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+6],
                y_vars=['SalePrice'])


# Most of the features appears to have a relationship with the SalePrice. 
# 
# The features which aren't found in the house are can be seen at the x=0 location, as can be seen.

# In[11]:


individual_features = []
for i in range(0, len(df_num.columns) - 1):
    ftr = df_num[[df_num.columns[i], 'SalePrice']]
    ftr = ftr[ftr[df_num.columns[i]] != 0]
    individual_features.append(ftr)


# In[160]:


all_corrs = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features}
for key, value in all_corrs.items():
    if value > 0.5:
        print(key, value)
        
print('')        
golden_features = [key for key, value in all_corrs.items() if value >= 0.5]


# When we deleted the 0 values, we found another feature with a positive correlation. Now we have 11 features in our golden_features. 

# # Heteroscedasticity and Removing

# In[46]:


def het_viz(df, feature):
    fig = px.scatter(df, x=feature, y="SalePrice")
    fig.show()


# In[48]:


for i in range(len(golden_features)):
    het_viz(df_num, str(golden_features[i]))


# When looked at the plots, it is evident that the features are heteroscedastic. Let's test it statistically and analyze it further.

# Breusch-Pagan test uses the following null and alternative hypotheses:
# 
# 
# H0: Homoscedasticity is present (residuals are equally scattered)
# 
# H1: Heteroscedasticity is present (residuals are not equally scattered)

# The null hypothesis is rejected if the p-value of the LM test is less than the significance level. This means that our data are heteroscedastic. And OLS model is no longer B.L.U.E(best linear unbiased estimator). That means error variance is biased, t and F statistics are invalid.This problem can be corrected with the Log Transform and Box-Cox Transform methods. Let's test this test for all features one by one. First of all we need to convert features into a 2d array since this is required for the input in Breusch-Pagan test. 

# In[68]:


def transform_2d_array(col):
    s = []
    for i in col:
        a = [1,i]
        s.append(a)
    return (np.array(s))


# In[110]:


def transform_features_to_2d_array(df, feature): 
    feature_model = transform_2d_array(df[str(feature)])
    return feature_model


# In[112]:


for i in range(len(golden_features)):
    bp_test = het_breuschpagan(model.resid, transform_features_to_2d_array(df_num, golden_features[i]))
    print(golden_features[i])
    print ('LM-test p_value')
    print (bp_test[1])
    print(' ')


# The p-values for all features are less than the significance level which is 0.05. So the null hypothesis is rejected for all features. Let's begin using the Log-Transform method to fix this.

# In[147]:


df_num['log_OverallQual'] = np.log(df_num['OverallQual'])
f ='log_OverallQual~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
log_OverallQual_model = transform_2d_array(df_num['log_OverallQual'])
bp_test = het_breuschpagan(df_model.resid, log_OverallQual_model)
print('log_OverallQual')
print('LM-test p_value')
print(bp_test[1])
print(' ')


df_num['log_YearBuilt'] = np.log(df_num['YearBuilt'])
f ='log_YearBuilt~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
log_YearBuilt_model = transform_2d_array(df_num['log_YearBuilt'])
bp_test = het_breuschpagan(df_model.resid, log_YearBuilt_model)
print('log_YearBuilt')
print('LM-test p_value')
print(bp_test[1])
print(' ')


df_num['log_YearRemodAdd'] = np.log(df_num['YearRemodAdd'])
f ='log_YearRemodAdd~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
log_YearRemodAdd_model = transform_2d_array(df_num['log_YearRemodAdd'])
bp_test = het_breuschpagan(df_model.resid, log_YearRemodAdd_model)
print('log_YearRemodAdd')
print('LM-test p_value')
print(bp_test[1])
print(' ')

df_num['log_1stFlrSF'] = np.log(df_num['1stFlrSF'])
f ='log_1stFlrSF~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
log_1stFlrSF_model = transform_2d_array(df_num['log_1stFlrSF'])
bp_test = het_breuschpagan(df_model.resid, log_1stFlrSF_model)
print('log_1stFlrSF')
print('LM-test p_value')
print(bp_test[1])
print(' ')

df_num['log_GrLivArea'] = np.log(df_num['GrLivArea'])
f ='log_GrLivArea~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
log_GrLivArea_model = transform_2d_array(df_num['log_GrLivArea'])
bp_test = het_breuschpagan(df_model.resid, log_GrLivArea_model)
print('log_GrLivArea')
print('LM-test p_value')
print(bp_test[1])
print(' ')

df_num['log_TotRmsAbvGrd'] = np.log(df_num['TotRmsAbvGrd'])
f ='log_TotRmsAbvGrd~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
log_TotRmsAbvGrd_model = transform_2d_array(df_num['log_TotRmsAbvGrd'])
bp_test = het_breuschpagan(df_model.resid, log_TotRmsAbvGrd_model)
print('log_TotRmsAbvGrd')
print('LM-test p_value')
print(bp_test[1])
print(' ')


# For 'TotalBsmtSF', '2ndFlrSF', 'FullBath', 'GarageCars', 'GarageArea' features mean is -inf and standard deviation is nan. ![Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202022-02-18%20135214.png](attachment:Ekran%20g%C3%B6r%C3%BCnt%C3%BCs%C3%BC%202022-02-18%20135214.png)
# 
# I won't go into detail about how to fix it right now because this project was created primarily for reviewing purposes.
# 
# Now our features are 'OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd'.

# Except for the '1stFlrSF' feature, we can see that the Log transform method couldn't not fix the heteroscedasticity problem. Let's try Box Cox transform.

# In[153]:


x, _ = stats.boxcox(df_num['OverallQual'])
df_num['trans_OverallQual'] = x
f ='trans_OverallQual~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
trans_OverallQual_model = transform_2d_array(df_num['trans_OverallQual'])
bp_test = het_breuschpagan(df_model.resid, trans_OverallQual_model)
print('trans_OverallQual')
print ('LM-test p_value')
print (bp_test[1])
print('')

x, _ = stats.boxcox(df_num['YearBuilt'])
df_num['trans_YearBuilt'] = x
f ='trans_YearBuilt~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
trans_YearBuilt_model = transform_2d_array(df_num['trans_YearBuilt'])
bp_test = het_breuschpagan(df_model.resid, trans_YearBuilt_model)
print('trans_YearBuilt')
print ('LM-test p_value')
print (bp_test[1])
print('')

x, _ = stats.boxcox(df_num['YearRemodAdd'])
df_num['trans_YearRemodAdd'] = x
f ='trans_YearRemodAdd~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
trans_YearRemodAdd_model = transform_2d_array(df_num['trans_YearRemodAdd'])
bp_test = het_breuschpagan(df_model.resid, trans_YearRemodAdd_model)
print('trans_YearRemodAdd')
print ('LM-test p_value')
print (bp_test[1])
print('')

x, _ = stats.boxcox(df_num['1stFlrSF'])
df_num['trans_1stFlrSF'] = x
f ='trans_1stFlrSF~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
trans_1stFlrSF_model = transform_2d_array(df_num['trans_1stFlrSF'])
bp_test = het_breuschpagan(df_model.resid, trans_1stFlrSF_model)
print('trans_1stFlrSF')
print ('LM-test p_value')
print (bp_test[1])
print('')

x, _ = stats.boxcox(df_num['GrLivArea'])
df_num['trans_GrLivArea'] = x
f ='trans_GrLivArea~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
trans_GrLivArea_model = transform_2d_array(df_num['trans_GrLivArea'])
bp_test = het_breuschpagan(df_model.resid, trans_GrLivArea_model)
print('trans_GrLivArea')
print ('LM-test p_value')
print (bp_test[1])
print('')

x, _ = stats.boxcox(df_num['TotRmsAbvGrd'])
df_num['trans_TotRmsAbvGrd'] = x
f ='trans_TotRmsAbvGrd~SalePrice'
df_model = ols(formula=f, data=df_num).fit()
trans_TotRmsAbvGrd_model = transform_2d_array(df_num['trans_TotRmsAbvGrd'])
bp_test = het_breuschpagan(df_model.resid, trans_TotRmsAbvGrd_model)
print('trans_TotRmsAbvGrd')
print ('LM-test p_value')
print (bp_test[1])


# It can be seen from the above result that p-value > significance level. This means that the heteroscedasticity is removed for some features. 
# 
# The Box-Cox transform is better to the Log transform when it comes to removing the heteroscedasticity.
# 
# However, besides from its ability to successfully remove heteroscedasticity, this technique has a lot of disadvantages, including:
# - Negative data cannot be used using this method.
# - If we only use the Log transform method, it might be easier to analyze our data.
