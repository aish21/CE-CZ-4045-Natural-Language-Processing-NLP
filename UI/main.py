#-------------------------------------------------------------------------#
#Imports
from tkinter import Y
import streamlit as st
import pandas as pd
import chardet
import altair as alt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
#import gensim
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models as gensimvis
#-------------------------------------------------------------------------#
#Config for the page
st.set_page_config(
    page_title = "NLP Grp 31 Data Visualisation",
    page_icon = None, #can change this later to favicon or even any emoji
)

alt.themes.enable("streamlit")
#-------------------------------------------------------------------------#
#GETTING CHARSET
with open('../data/nlp_vader_textblob_classified_data.csv', 'rb') as f:
    f.readline() 
    enc = chardet.detect(f.readline())  # or readline if the file is large
    #f.read() was taking too long and f.readline() didnt work because 
    #i assume the first line was giving the wrong encoding for whatever 
    #reason but getting the encoding from line 2 just worked idk why

#-------------------------------------------------------------------------#
#Team selection widget
st.subheader("Choose the teams")  

choices = st.multiselect(
        label = "Select a team",
        options = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "Manchester United", "Tottenham"],
        default = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "Manchester United", "Tottenham"],
        help = "Choose any combination of the teams you want to see",
        label_visibility = "collapsed"
        )                
#-------------------------------------------------------------------------#
#Loading of team data with a nice spinner
with st.spinner(text = "Loading team data"):
    tweetData = pd.read_csv('../data/nlp_vader_textblob_classified_data.csv', encoding = enc["encoding"], index_col=False)
    
    df = tweetData.loc[tweetData['primaryTeam'].isin(choices)] 
    #Choose only the data of the teams we want

#-------------------------------------------------------------------------#
#just to display the dataframe
#with st.spinner(text = "Loading dataframe"):
#    st.title("The Data")
#    st.dataframe(data = df)
#THIS TAKES THE LONGEST TIME - REMOVE??
#-------------------------------------------------------------------------#
#Ratio of teams
with st.spinner(text = "Loading Ratio"):
    st.title("Ratio of tweets by team")
    c = alt.Chart(df).mark_bar().encode(
            y = alt.Y("primaryTeam:N"), 
            x = "count():Q",
            color = "primaryTeam:N"
            ).interactive()
    st.altair_chart(c, use_container_width = True)

#-------------------------------------------------------------------------#
#Show histogram of length of tweets
with st.spinner(text = "Loading histogram"):
    st.title("Length of the Tweets")
    df["Length in Characters"] = df['content'].str.len()
    c = alt.Chart(df).mark_bar().encode(
            x = alt.X("Length in Characters:Q", bin = alt.Bin(maxbins=80)), 
            y = "count():Q",
            color = "primaryTeam:N"
            ).interactive()
    st.altair_chart(c, use_container_width = True)

    df["Length in Words"] = df['content'].str.split().map(lambda x: len(x))

    c = alt.Chart(df).mark_bar().encode(
            x = alt.X("Length in Words:Q", bin = False),
            y = "count()",
            color = "primaryTeam:N"
            ).interactive()
    st.altair_chart(c, use_container_width = True)

#-------------------------------------------------------------------------#
#Common Stopwords

with st.spinner(text = "Loading stopwords"):
    st.title("Stopwords")

    

    corpus=[]
    check= tweetData['content'].str.split()
    check=check.values.tolist()
    corpus=[word for i in check for word in i]
    stop=set(stopwords.words('english'))
    from collections import defaultdict
    dic=defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word]+=1

    n = st.slider("How many top words?", 5, 25, 10)  
    top = sorted(dic.items(), key=lambda x:x[1],reverse=True)[:n]
    top = pd.DataFrame(top, columns = ["Words", "Counts"])

    c = alt.Chart(top).mark_bar().encode(
            y = alt.Y("Words:N", sort="-x"),
            x = "Counts:Q",
            color = "Words:N"
            ).interactive()
    st.altair_chart(c, use_container_width = True)

#-------------------------------------------------------------------------#
#Top k n-grams
    
with st.spinner(text = "Loading ngrams"):
    st.title("n grams")
    n = st.slider("n-gram", 2, 5, 2) 
    k = st.slider("How many n-grams?", 5, 25, 10) 

    def top_ngrams(corpus, n):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_k_ngrams = top_ngrams(df['content'],n)[:k]
    print(top_k_ngrams)
    top = pd.DataFrame(top_k_ngrams, columns = [f"ngram", "Counts"])
    c = alt.Chart(top).mark_bar().encode(
            y = alt.Y("ngram:N", sort="-x"),
            x = "Counts:Q",
            color = "ngram:N"
            ).interactive()
    st.altair_chart(c, use_container_width = True)



def get_lda_objects(text):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop=set(stopwords.words('english'))

    
    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for news in text:
            words=[w for w in word_tokenize(news) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 10, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
    
    return lda_model, bow_corpus, dic

def plot_lda_vis(lda_model, bow_corpus, dic):
    vis = gensimvis.prepare(lda_model, bow_corpus, dic)
    return vis

lda_model, bow_corpus, dic = get_lda_objects(tweetData['content'])

lda_model.show_topics()

vis = plot_lda_vis(lda_model, bow_corpus, dic)

st.altair_chart(vis, use_container_width = True)