import streamlit as st
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from requirefunction import sent_to_words,remove_stopwords,load_data,sentiment_pred,lda_top #lda_model_fun
from  matplotlib.ticker import FuncFormatter 

st.set_page_config(layout="wide")
# st.subheader("New Innovation From Rasith")

choice = st.sidebar.selectbox("Select your choice", ["On Text", "On CSV"])

if choice == "On Text":
    st.markdown("<h1 style='text-align: center; color: red;'>FeedbackAnalyzerAI 🕵️‍♂️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Predict sentiment 😄|🙂|😈 of your review [Text]</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: black;'> ©rasithbm </p>", unsafe_allow_html=True)
    # st.button("Click Me 👈")
    
    # Create a text area widget to allow users to paste transcripts
    text_input = st.text_area("Paste enter text below", height=100)
    if text_input is not None:
       if st.button("Analyze Text"):
          label_pred = sentiment_pred(text_input).upper()
          st.text_area(label="Sentiment Prediction", value=label_pred, height=40)
          st.markdown("<h4 style='text-align: center; color: black;'> ©rasithbm </h4>", unsafe_allow_html=True)
          
else:
    st.markdown("<h1 style='text-align: center; color: red;'>FeedbackAnalyzerAI 🕵️‍♂️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>Predict sentiment 😄|🙂|😈 of your review [CSV/Excel]</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: black;'> ©rasithbm </p>", unsafe_allow_html=True)

    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    def call_back():
        st.session_state.button_clicked = True

    upload_csv = st.file_uploader("Upload your CSV / Excel file", type=['csv','xlx'])
    if upload_csv is not None:
        if (st.button("Analyze CSV File",disabled=st.session_state.button_clicked) or st.session_state.button_clicked):
            # dff = pd.read_csv(upload_csv, encoding= 'unicode_escape')
            dff = load_data(upload_csv)
           
            rev_li = list(dff[dff.columns[0]])
            label=[]
            for review in rev_li:
                output=sentiment_pred(review) #("I like you I love you")
                label.append(output)
            dff['labels']=pd.DataFrame({'extrc_label':label})
            st.dataframe(dff)
            data=dff.to_csv(index=False).encode('utf-8')
            st.download_button("Download data as CSV",
                               data,
                               "text/csv",
                               key="Predicted_reviews.csv",)

            # Avoid error in streamlit app
            st.set_option('deprecation.showPyplotGlobalUse', False)

            # Make Bar chart to visualize sentiment of the app
            st.subheader("Sentiment Visual")
            plt.figure(figsize=(3, 2))
            ax=sns.countplot(data=dff, x='labels')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
            bar=plt.show()
            st.pyplot(bar)

            if st.sidebar.button("Analyze Positive Reviews",on_click=call_back):
                positive_re = dff[dff['labels']=='positive']
                topics = lda_top(positive_re)
                try:
                    st.subheader("Postive Reviews")
                    st.dataframe(positive_re)
                    data_p=positive_re.to_csv(index=False).encode('utf-8')
                    st.download_button("Download data as CSV",
                                    data_p, 
                                    "Predicted_Positive_reviews.csv",
                                    "text/csv",
                                    key="download-tools-csv",)

                    for t in range(topics.num_topics):
                        plt.figure()
                        plt.imshow(WordCloud().fit_words(dict(topics.show_topic(t, 200))))
                        plt.axis("off")
                        plt.title("Topic #" + str(t))
                        fig = plt.show()
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot(fig)

                except:
                    st.markdown("No Positive Reviews")

            if st.sidebar.button("Analyze Negative Reviews",on_click=call_back):
                negative_re = dff[dff['labels']=='negative']
                topics_ne = lda_top(negative_re)
                try:
                    st.subheader("Negative Reviews")
                    st.dataframe(negative_re)
                    data_n=negative_re.to_csv(index=False).encode('utf-8')
                    st.download_button("Download data as CSV",
                                    data_n,
                                    "Predicted_Negative_reviews.csv",
                                    "text/csv",
                                    key="download-tools-csv",)

                    for t in range(topics_ne.num_topics):
                        plt.figure()
                        plt.imshow(WordCloud().fit_words(dict(topics_ne.show_topic(t, 200))))
                        plt.axis("off")
                        plt.title("Topic #" + str(t))
                        fig = plt.show()
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        st.pyplot(fig)
                except:
                    st.markdown("No Negative Reviews")
        




