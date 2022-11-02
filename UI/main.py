#-------------------------------------------------------------------------#
#Imports
from tkinter import Y
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import chardet
import altair as alt
import nltk
import ssl
# In case there are issues while downloading stopwrods due to ssl licence/certificate
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
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
st.title('Premier League Tweets Sentiment Analysis')
#-------------------------------------------------------------------------#
#GETTING CHARSET
with open('../data/nlp_vader_textblob_classified_data.csv', 'rb') as f:
    f.readline() 
    enc = chardet.detect(f.readline())  
    # or readline if the file is large
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
        label_visibility = "collapsed")                
#-------------------------------------------------------------------------#
#Loading of team data with a nice spinner
with st.spinner(text = "Loading team data"):
    tweetData = pd.read_csv('../data/nlp_vader_textblob_classified_data.csv', encoding = enc["encoding"], index_col=False)
    
    df = tweetData.loc[tweetData['primaryTeam'].isin(choices)] 
    #Choose only the data of the teams we want

#-------------------------------------------------------------------------#
#just to display the dataframe
with st.spinner(text = "Loading dataframe"):
   st.title("Dataset Sample")
   st.dataframe(data = df[:10])
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

    top_k_ngrams = top_ngrams(df['vader_preprocessing_text'],n)[:k]
    print(top_k_ngrams)
    top = pd.DataFrame(top_k_ngrams, columns = [f"ngram", "Counts"])
    c = alt.Chart(top).mark_bar().encode(
            y = alt.Y("ngram:N", sort="-x"),
            x = "Counts:Q",
            color = "ngram:N"
            ).interactive()
    st.altair_chart(c, use_container_width = True)

#-------------------------------------------------------------------------#
# Sentiment pie Chart
with st.spinner(text = "Loading Sentiment Pie Chart"):
    st.title("Tweet Sentiment Pie Chart")
    pc = px.pie(df,values = df['final_class'].value_counts(normalize=True),names = ['Negative','Positive','Neutral'],color = df['final_class'].unique()
            ,color_discrete_map={-1:'red',
                                 1:'green',
                                 0:'blue'})
    st.plotly_chart(pc)

#-------------------------------------------------------------------------#
# Map Data
manU_map = "man_u_map.html"
manC_map = "ManC_map.html"
chelsea_map = "chelsea_map.html"
liv_map = "liverpool_map.html"
arsenal_map = "arsenal_map.html"
tot_map = "tot_map.html"

with st.spinner(text = "Loading tweet location data"):
    st.title("Tweet Locations")

    # Read file and keep in variable
    with open(manU_map,'r') as f: 
        manU_data = f.read()
    
    with open(manC_map,'r') as f: 
        manC_data = f.read()
    
    with open(chelsea_map,'r') as f: 
        chelsea_data = f.read()
    
    with open(liv_map,'r') as f: 
        liv_data = f.read()
    
    with open(arsenal_map,'r') as f: 
        arsenal_data = f.read()
    
    with open(tot_map,'r') as f: 
        tot_data = f.read()

    # Show in maps
    st.header("Manchester United")
    st.components.v1.html(manU_data, height=400, width=700)

    # Show in maps
    st.header("Manchester City")
    st.components.v1.html(manC_data, height=400, width=700) 

    # Show in maps
    st.header("Chelsea")
    st.components.v1.html(chelsea_data, height=400, width=700) 

    # Show in maps
    st.header("Liverpool")
    st.components.v1.html(liv_data, height=400, width=700) 

    # Show in maps
    st.header("Arsenal")
    st.components.v1.html(arsenal_data, height=400, width=700) 

    # Show in maps
    st.header("Tottenham")
    st.components.v1.html(tot_data, height=400, width=700) 

#------------------------------------------------------------#
# Results
with st.spinner(text = "Loading Evaluation Metrics"):
   st.title("Summary of Evaluation Metrics")
data= {1: ['Gaussian Naive Bayes',0.469,0.473,0.482,0.4815],
                          2:['Random Forest',0.445, 0.456, 0.445, 0.445],
                          3: ['VADER',0.460,0.469,0.461,0.4611],
                          4: ['TextBlob',0.446,0.462,0.442,0.4422],
                          5:['LSTM+Attention Neural net',],
                          6: ['RoBERTa Model(with Enhancements)',0.78,0.76,0.76,0.78]}
evaldf = pd.DataFrame.from_dict(data,orient = 'index',
   columns= ['Models','F1','Precision','Recall','Accuracy']) 
st.dataframe(data = evaldf)

