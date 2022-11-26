#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import date
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config


# In[45]:



user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10


# In[46]:


df1 = pd.DataFrame()
company_name = input("Please provide the name of the Company or a Ticker: ")
#As long as the company name is valid, not empty...
if company_name != '':
    print(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')
    
for i in range(30,60,2):
    now = dt.date.today() - dt.timedelta(days = i)
    now = now.strftime('%m-%d-%Y')
    yesterday = dt.date.today() - dt.timedelta(days = i+1)
    yesterday = yesterday.strftime('%m-%d-%Y')
    #Extract News with Google News
    googlenews = GoogleNews(start=yesterday,end=now)
    googlenews.search(company_name)
    result = googlenews.result()
    #store the results
    df = pd.DataFrame(result)
    df1=df1.append(df)
    
print(df1)


# In[47]:


try:
    list =[] #creating an empty list 
    for i in df1.index:
        dict = {} #creating an empty dictionary to append an article in every single iteration
        article = Article(df['link'][i],config=config) #providing the link
        try:
          article.download() #downloading the article 
          article.parse() #parsing the article
          article.nlp() #performing natural language processing (nlp)
        except:
           pass 
        #storing results in our empty dictionary
    
        dict['Datetime']=df1['datetime'][i] 
        dict['Media']=df1['media'][i]
        dict['Title']=article.title
        dict['Article']=article.text
        dict['Summary']=article.summary
        dict['Key_words']=article.keywords
        
        list.append(dict)
    check_empty = not any(list)
    # print(check_empty)
    if check_empty == False:
      news_df=pd.DataFrame(list) #creating dataframe
      print(news_df)

except Exception as e:
    #exception handling
    print("exception occurred:" + str(e))
    print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.' )
    


# In[48]:


df1.head()


# In[49]:


df2=df1.set_index(df1['datetime'])


# In[50]:


df2=df2['title']
df2.head()


# In[ ]:




