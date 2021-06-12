import streamlit as st
import numpy as np
import re

st.title("NLP demo")
st.write("Demo working of BOW and TFIDF")

corpus = st.text_input('paragraph','',help='Enter your paragraph here')
list_of_sent = []
# %%
def sent_token(para):
    '''
    takes para as input
    returns list of sentences
    '''
    pattern = r'[\.\!\?]'
    para = re.sub(pattern,'|' ,para)
    sent = para.split('|')
    sent = list(map(lambda x:str.strip(x),sent))
    sent = list(filter(lambda x:x!='',sent))
    return sent
# sent_token('this pasta is Very tasty. This pastA is good! This pasta is miserable. Pasta is pasta?')
# %%
def sent_clean(sent_tokens):
    sent_list = []
    for sent in sent_tokens:
        sent_cleaned = sent.lower()
        pattern = r'[^A-Za-z\s]'  #to keep numbers r'[^\w\s]
        sent_cleaned = re.sub(pattern, '',sent_cleaned)
        sent_list.append(sent_cleaned)
    return sent_list

# %%
def BOW(sent_token):
    '''
    
    '''
    pass

#%%
if st.button('Calculate'):
    sent_tokens =  sent_token(corpus) 
    list_of_sent = sent_clean(sent_tokens)
    st.write(list_of_sent)
    bow_array = BOW(list_of_sent)