# Do everything in tweet analysis.
# do it for every individual team also

import streamlit as st
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import altair as alt

fig, ax = plt.subplots()

with open('../data/nlp_vader_textblob_classified_data.csv', 'rb') as f:
    f.readline() 
    enc = chardet.detect(f.readline())  # or readline if the file is large

    #f.read() was taking too long and f.readline() didnt work because 
    #i assume the first line was giving the wrong encoding for whatever 
    #reason but getting the encoding from line 2 just worked idk why
    
tweetData = pd.read_csv('../data/nlp_vader_textblob_classified_data.csv', encoding = enc["encoding"], index_col=False)

st.title("the Data")
st.dataframe(data = tweetData)

st.title("Length of the Tweets")
tweetData["charlen"]= tweetData['content'].str.len()
c = alt.Chart(tweetData).mark_bar().encode(alt.X("charlen:Q", bin = True), y = "count()")
st.altair_chart(c) #matplotlib charts look v ugly


st.button(label = "Balloons!", on_click = st.balloons)