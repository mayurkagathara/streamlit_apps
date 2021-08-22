import streamlit as st
import pandas as pd
import re

st.title("NLP demo for BOW")
st.write("Demo working of BOW")

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
def BOW(list_of_sent):
    '''
    input is list of sentence
    output is list of vectorized sentence
    '''
    idx = 0
    sent_word_count = dict()
    set_of_words = set()
    for sent in list_of_sent:
        sent_word_count[idx] = dict()
        for word in sent.split():
            set_of_words.add(word)
            sent_word_count[idx][word] = sent_word_count[idx].get(word,0) + 1
        idx += 1
    print(sent_word_count)

    list_of_words = sorted(list(set_of_words))
    print(list_of_words)

    Bag_of_words_mat = []
    # Bag_of_words_mat.append(list_of_words)

    for i in range(len(list_of_sent)):
        sent_vec = []
        for word in list_of_words:
            sent_vec.append(sent_word_count[i].get(word,0))
        Bag_of_words_mat.append(sent_vec)
    
    print(Bag_of_words_mat)
    df = pd.DataFrame(Bag_of_words_mat,columns=list_of_words)
    return df
#%%
if st.button('Calculate'):
    sent_tokens =  sent_token(corpus) 
    list_of_sent = sent_clean(sent_tokens)
    st.write(list_of_sent)
    bow_array = BOW(list_of_sent)
    st.title('Bag Of Words(BOW)')
    st.table(bow_array)