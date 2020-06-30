#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install gtts


# In[1]:


from gtts import gTTS      #google text to speech api
import os


# In[7]:


text = ' Hello, My name is Anju. How are you?'
language = 'en'
speech = gTTS(text=text,
             lang = language,
             slow = False)
speech.save('Hello.mp3')
os.system('start Hello.mp3')


# In[ ]:


# you can go to the root folder to check your mp3.

