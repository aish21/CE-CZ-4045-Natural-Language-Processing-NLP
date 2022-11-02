from gensim import corpora as gensimcorpora

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk

from gensim import models as gensimmodels
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
import pyLDAvis.gensim_models as gensimvis
import pandas as pd
import chardet


with open('../data/nlp_vader_textblob_classified_data.csv', 'rb') as f:
    f.readline() 
    enc = chardet.detect(f.readline()) 

    
tweetData = pd.read_csv('../data/nlp_vader_textblob_classified_data.csv', encoding = enc["encoding"], index_col=False)
    
