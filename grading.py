#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
pd.set_option('display.max_columns', 100) 


# In[2]:


df = pd.read_csv("grading_data.csv")
df.head(3)


# In[3]:


df.shape


# In[4]:


df["Category"] = df.category.replace({1:"SC", 2:"ST",3:"OBC",4:"EBC",5:"GEN",6:"others"})


# In[5]:


df["Education"] = df.education.replace({1:"8th",2:"10th",3:"12th",4:"Graduate",5:"Post graduate"})


# In[6]:


from sklearn.preprocessing import LabelEncoder 
le =LabelEncoder()


# In[57]:


dfle = df
dfle['overall grades'] = le.fit_transform(dfle['overall grade'])
dfle['districts'] = le.fit_transform(dfle['Districts'])
dfle.head(5)


# In[8]:


dfle.describe()


# In[9]:


correlation = round(dfle.corr(),2)


# In[10]:


plt.figure(figsize=(12,12))
sns.heatmap(correlation,annot =True)
plt.savefig("cor.pdf")


# In[11]:


##sns.pairplot(df)


# In[12]:


## number of pashu sakhi sampled from every district
plt.figure(figsize=(8,6))
sns.set_theme(style="darkgrid")
sns.countplot(dfle["Districts"])
plt.xlabel('Districts')
plt.ylabel('Number of Pashu Sakhi')
plt.savefig("number.pdf")


# In[13]:


## Educational profile of the pashu sakhi 
plt.figure(figsize=(8,6))
sns.set_theme(style="darkgrid")
sns.countplot(dfle["Education"])
plt.xlabel('Educational Qualification')
plt.ylabel('Number of Pashu Sakhi')
plt.savefig("education.pdf")


# In[14]:


## Break up of educational qualification 
plt.figure(figsize=(10,8))
sns.set_theme(style="darkgrid")
sns.countplot(x="Districts", hue="Education", data=dfle)
plt.xlabel("Educational Qualification")
plt.ylabel("Number of pashu sakhi")
plt.savefig("district.pdf")


# In[15]:


## Comparison of distribution of marks scored by pashu sakhis in every districts 
## We can see the difference among the districbution of performances of pashu sakhi across the districts 
plt.figure(figsize=(8,6))
sns.boxplot(x=dfle['Districts'], y=dfle['weighted average'])
plt.savefig("score.pdf")


# In[16]:


## ANOVA test on districts as an independent variable 


# In[17]:


## comparison of distribution of weighted average marks scored according to educational qualification 
plt.figure(figsize=(8,6))
sns.set_theme(style="white", palette=None)
sns.boxplot(x=dfle['Education'],y=dfle['weighted average'])
plt.savefig("marks.pdf")


# In[18]:


plt.figure(figsize=(8,5))
sns.boxplot(x=dfle['Category'], y=dfle['weighted average'])
plt.savefig("category.pdf")


# In[19]:


## anova test on education as an independent variable 


# In[20]:


## distribution of the age of the pashu sakhis who were part of the study 
sns.set_theme(style="darkgrid")
sns.histplot(dfle["age"],bins=8)
plt.xlabel("Age")
plt.ylabel("Number of Pashu Sakhi")
plt.savefig("age.pdf")


# In[21]:


## count of pashu sakhi according to their grades
plt.figure(figsize=(8,6))
sns.set_theme(style="darkgrid")
sns.countplot(dfle["overall grade"])
plt.xlabel('Grades')
plt.ylabel('Number of pashu sakhi')
plt.savefig("gradescount.pdf")


# In[22]:


## pashu sakhi according to their grades in different districts
plt.figure(figsize=(10,8))
sns.set_theme(style="darkgrid")
sns.countplot(x="Districts", hue="overall grade", data=dfle)
plt.savefig("gradecountdistrictwise.pdf")


# In[23]:


## use of cross tab to look at the relationship between overall grade and educational attainment 
cf = pd.crosstab(dfle["Education"],dfle["overall grade"])
cf


# In[24]:


df.groupby(["Districts","Education","overall grade"]).size().unstack().fillna(0)


# In[25]:


dfle["districts"].unique()


# In[26]:


## detailed analysis of purnea district
pf = dfle[dfle.districts == 5]                     ## Purnea
zf = dfle[dfle.districts==2]                       ## Muzaffarpur 
nf =  dfle[dfle.districts==3]                      ## Nalanda
rf =   dfle[dfle.districts==6]                     ## Rohtas
gf =  dfle[dfle.districts==1]                      ## Gaya
na =  dfle[dfle.districts==0]                      ## Araria
pf.head(3)


# In[27]:


plt.figure(figsize=(10,8))
sns.set_theme(style="darkgrid")
sns.countplot(x="block", hue="overall grades", data=pf)
plt.xlabel("Blocks")
plt.ylabel("Number Of Pashu Sakhi")


# In[28]:


pf["block"].unique()


# In[29]:


## Hypothesis testing 


# In[30]:


rf.head()


# In[31]:


## testing the hypothesis 
## mean of atleast one distict is different from other 
meanP = pf["weighted average"].mean()
meanZ= zf["weighted average"].mean()
meanN = nf["weighted average"].mean()
meanR = nf["weighted average"].mean()
meanA = gf["weighted average"].mean()
meanG = rf["weighted average"].mean()


# In[32]:


means = [meanP,meanZ,meanN,meanR,meanA,meanG]
means


# In[33]:


import scipy.stats as stats 
from statsmodels.formula.api import ols
import statsmodels.api as sm 


# In[34]:


model = ols('age~districts', data = dfle).fit()


# In[35]:


annova_result = sm.stats.anova_lm(model,typ=1)
print(annova_result)


# In[36]:


stats.f_oneway(dfle["age"],dfle["districts"])


# In[63]:


dfle = dfle.rename(columns={'weighted average': 'weighted_average'})
dfle.head(3)


# In[64]:


stats.f_oneway(dfle['weighted_average'],dfle['districts'])
## F Statistic is really high and Pvalue is very small(due to e raise to the power negative number ) so we can conclude
## that the mean of weighted average differs across the districts


# In[39]:


model = ols('weighted_average~districts', data = dfle).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)
## Here p value is significant(less than 0.05)hence mean of atleast one of the district is different from other 


# In[40]:


import scipy.stats as stats
stats.shapiro(model.resid)
## Null hypothesis: data is drawn from normal distribution.
## the test indicates that the output is nonsiginficant hence weighted average score is normally distributed 
## As P value is large we fail to reject the null hypthesis and assume that residuals are normally distributed 


# In[41]:


stat = stats.probplot(model.resid, plot= plt, rvalue= True)
## it is reasonable to assume that assumption of normality was met 
## As the standardized residuals lie around the 45-degree line, it suggests that the residuals are approximately normally distributed


# In[42]:


plt.hist(model.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()


# In[43]:


## Null hypothesis - weighted average has equal variance
from scipy.stats import levene
stat, p = levene(pf["weighted average"],zf["weighted average"],nf["weighted average"],na["weighted average"])
print(p)
## P value is too high hence we cant reject the null hypothesis 


# In[44]:


## perform post-hoc test 
## to know which one of the pairs in the group is causing the significant result 


# In[45]:


pip install bioinfokit


# In[46]:


import bioinfokit


# In[47]:


from bioinfokit.analys import stat
res = stat()


# In[52]:


##res.tukey_hsd(df=dfle, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)')
##res.tukey_summary


# In[49]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog =dfle["weighted_average"],groups = dfle["Districts"],alpha = 0.05 )
##plt.vlines(x=0.65,ymin= -0.5, ymax=4.5,color="red")
tukey.plot_simultaneous()
tukey.summary()
##plt.savefig("an.pdf")


# In[ ]:


## check the hypothesis if education has any impact on mean weighted score 


# In[51]:


model = ols('weighted_average~Education', data = dfle).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)
## As p value is very less hence the result is significant 


# In[65]:


stats.f_oneway(dfle['weighted_average'],dfle['education'])
## F statistics is large and pvalue is small 


# In[ ]:


## Tukey test to see which pair has caused such a small P value 


# In[66]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog =dfle["weighted_average"],groups = dfle["Education"],alpha = 0.05 )
tukey.plot_simultaneous()
tukey.summary()


# In[ ]:


## graduates have not scored significantly higher than those who have studied till class 8th 


# In[58]:


## checking for the impact of category on weighted_average 


# In[67]:


model = ols('weighted_average~category', data = dfle).fit()
anova_result = sm.stats.anova_lm(model, typ=2)
print(anova_result)
## The impact of category on weighted average score is not statistically significant as p value is large (greater than 0.05)


# In[ ]:




