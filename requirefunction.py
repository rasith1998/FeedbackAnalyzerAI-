import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import torch 
from transformers import TFAutoModelForSequenceClassification,AutoTokenizer, AutoConfig
from scipy.special import softmax
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation,strip_numeric
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import gensim.corpora as corpora

@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding= 'unicode_escape')
    return df

@st.cache_resource
def sentiment_pred(text):
    # For sentiment prediction
    MODEL_Path = "Rasith/NzsentimentApp" 
    # MODEL_Path = "F:/Review_Senti_App/saved_model/NZapp_model"
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_Path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_Path)
    config = AutoConfig.from_pretrained(MODEL_Path)

    input=tokenizer(text,return_tensors='tf')
    output = config.id2label[np.argmax(softmax(model(input)[0][0].numpy()))]

    return output

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

#@st.cache
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

@st.cache
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

@st.cache
def lda_top(data):
    data1 = data.iloc[:, 0].values.tolist()
    data_words = list(sent_to_words(data1))
    # remove stop words
    data_words = remove_stopwords(data_words)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
     # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # number of topics
    num_topics = 5 # topics_pos
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)
    return lda_model