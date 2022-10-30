#-------------------------------------------------------------------------#
#Imports
from tkinter import Y
import streamlit as st
import pandas as pd
import chardet
import altair as alt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image

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

#-------------------------------------------------------------------------#
# LDA data
manU_lda = "manu_lda.html"
manC_lda = "manc_lda.html"
chelsea_lda = "chelsea_lda.html"
liv_lda = "liv_lda.html"
arsenal_lda = "arsenal_lda.html"
tot_lda = "tot_lda.html"

with st.spinner(text = "Loading tweets LDA"):
    st.title("Latent Dirichlet Allocation")

    # Read file and keep in variable
    with open(manU_lda,'r') as f: 
        manU_lda_data = f.read()
    
    with open(manC_lda,'r') as f: 
        manC_lda_data = f.read()
    
    with open(chelsea_lda,'r') as f: 
        chelsea_lda_data = f.read()
    
    with open(liv_lda,'r') as f: 
        liv_lda_data = f.read()
    
    with open(arsenal_lda,'r') as f: 
        arsenal_lda_data = f.read()
    
    with open(tot_lda,'r') as f: 
        tot_lda_data = f.read()

    # Show in maps
    st.header("Manchester United")
    st.components.v1.html(manU_lda_data, height=800, width=800, scrolling=True)

    # Show in maps
    st.header("Manchester City")
    st.components.v1.html(manC_lda_data, height=800, width=800, scrolling=True) 

    # Show in maps
    st.header("Chelsea")
    st.components.v1.html(chelsea_lda_data, height=800, width=800, scrolling=True) 

    # Show in maps
    st.header("Liverpool")
    st.components.v1.html(liv_lda_data, height=800, width=800, scrolling=True) 

    # Show in maps
    st.header("Arsenal")
    st.components.v1.html(arsenal_lda_data, height=800, width=800, scrolling=True) 

    # Show in maps
    st.header("Tottenham")
    st.components.v1.html(tot_lda_data, height=800, width=800, scrolling=True)

#-------------------------------------------------------------------------#
# Word Cloud
wc = Image.open('wc.png')
wc_proc = Image.open('wc_proc.png')
manu_wc = Image.open('manu_wc.png')
manu_wc_proc = Image.open('manu_wc_proc.png')
manc_wc = Image.open('manc_wc.png')
manc_wc_proc = Image.open('manc_wc_proc.png')
arsenal_wc = Image.open('arsenal_wc.png')
arsenal_wc_proc = Image.open('arsenal_wc_proc.png')
chelsea_wc = Image.open('chelsea_wc.png')
chelsea_wc_proc = Image.open('chelsea_wc_proc.png')
liv_wc = Image.open('liv_wc.png')
liv_wc_proc = Image.open('liv_wc_proc.png')
tot_wc = Image.open('tot_wc.png')
tot_wc_proc = Image.open('tot_wc_proc.png')

with st.spinner(text = "Loading WordCloud"):
    st.title("WordCloud - Raw Data")
    st.image(wc, caption='Wordcloud for raw corpus')
    st.image(manu_wc, caption='Wordcloud for raw corpus - Manchester United')
    st.image(manc_wc, caption='Wordcloud for raw corpus - Manchester City')
    st.image(arsenal_wc, caption='Wordcloud for raw corpus - Arsenal')
    st.image(chelsea_wc, caption='Wordcloud for raw corpus - Chelsea')
    st.image(liv_wc, caption='Wordcloud for raw corpus - Liverpool')
    st.image(tot_wc, caption='Wordcloud for raw corpus - Tottenham')

    st.title("WordCloud - Processed Corpus")
    st.image(wc_proc, caption='Wordcloud for processed corpus')
    st.image(manu_wc_proc, caption='Wordcloud for raw corpus - Manchester United')
    st.image(manc_wc_proc, caption='Wordcloud for raw corpus - Manchester City')
    st.image(arsenal_wc_proc, caption='Wordcloud for raw corpus - Arsenal')
    st.image(chelsea_wc_proc, caption='Wordcloud for raw corpus - Chelsea')
    st.image(liv_wc_proc, caption='Wordcloud for raw corpus - Liverpool')
    st.image(tot_wc_proc, caption='Wordcloud for raw corpus - Tottenham')