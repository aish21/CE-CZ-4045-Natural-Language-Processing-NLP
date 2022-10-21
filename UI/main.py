#-------------------------------------------------------------------------#
#Imports
import streamlit as st
import pandas as pd
import chardet
import altair as alt

#-------------------------------------------------------------------------#
#Config for the page
st.set_page_config(
    page_title = "NLP Grp 31 Data Visualisation",
    page_icon = None, #can change this later to favicon or even any emoji
)
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

def formatfunc(inp):
    if inp == "Tottenham":
        return "Tottenahm Hotspurs"
    if inp == "ManCity":
        return "Manchester City"
    return inp

choices = st.multiselect(
        label = "Select a team",
        options = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "Manchester United", "Tottenham"],
        default = ["Arsenal", "Chelsea", "Liverpool", "ManCity", "Manchester United", "Tottenham"],
        format_func = formatfunc,
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
with st.spinner(text = "Loading dataframe"):
    st.title("The Data")
    st.dataframe(data = df)
    
#-------------------------------------------------------------------------#
#Show histogram of length of tweets
with st.spinner(text = "Loading histogram"):
    st.title("Length of the Tweets")
    tweetData["charlen"]= tweetData['content'].str.len()
    c = alt.Chart(tweetData).mark_bar().encode(alt.X("charlen:Q", bin = True), y = "count()")
    st.altair_chart(c) #matplotlib charts look v ugly
