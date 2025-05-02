import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from collections import Counter
import streamlit as st

nltk.download(['punkt', 'stopwords'], quiet=True)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def load_sentiment_lexicon():
    positive_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
    negative_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"

    try:
        positive_lexicon = set(pd.read_csv(positive_url, sep="\t", header=None)[0])
    except Exception as e:
        st.warning(f"Gagal memuat positive.tsv dari URL: {e}")
        positive_lexicon = set()

    try:
        negative_lexicon = set(pd.read_csv(negative_url, sep="\t", header=None)[0])
    except Exception as e:
        st.warning(f"Gagal memuat negative.tsv dari URL: {e}")
        negative_lexicon = set()

    return positive_lexicon, negative_lexicon

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception:
        st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
        return None

def preprocess_data(data):
    required_columns = ['Date', 'Username', 'Rating', 'Reviews Text']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
        return None
    return data[required_columns].drop_duplicates(subset='Reviews Text', keep='first')

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()  # Case folding
    return re.sub(r'[^a-z\s]', '', text).strip()  # Cleaning

def normalize_text(text, norm_dict):
    if norm_dict:
        return ' '.join(norm_dict.get(word, word) for word in text.split())
    return text

def tokenize_text(text):
    if text:
        return word_tokenize(text)
    return []

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian')) - {'tidak', 'bukan'}
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def label_sentiment(tokens, positive_words, negative_words):
    positive_count = sum(1 for token in tokens if token in positive_words)
    negative_count = sum(1 for token in tokens if token in negative_words)
    sentiment_score = positive_count - negative_count

    if sentiment_score > 0:
        sentiment = "Positive"
    elif sentiment_score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment_score, sentiment

def preprocess_for_model(tokens):
    return ' '.join(tokens)

def generate_wordcloud(tokens_list, title, color):
    all_words = ' '.join(word for tokens in tokens_list for word in tokens)
    return WordCloud(width=800, height=400, background_color='white',
                     colormap=color, max_words=100).generate(all_words)

def generate_frequency_chart(tokens_list, top_n=10):
    all_words = [word for tokens in tokens_list for word in tokens]
    word_freq = Counter(all_words).most_common(top_n)
    words, freqs = zip(*word_freq)
    fig = go.Figure([go.Bar(x=words, y=freqs, text=freqs, textposition='auto',
                            marker_color='#4ECDC4', width=0.8)])
    fig.update_layout(title=f"Top {top_n} Kata Paling Sering Muncul",
                      xaxis_title="Kata", yaxis_title="Frekuensi",
                      title_x=0.5, height=450)
    return fig

def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions, labels=['Negative', 'Neutral', 'Positive'])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Negative', 'Neutral', 'Positive'],
                    y=['Negative', 'Neutral', 'Positive'])
    fig.update_layout(title=f"Confusion Matrix - {model_name}", title_x=0.5)
    return fig

def plot_data_split(train_size, test_size):
    fig = go.Figure([go.Bar(x=['Data Training', 'Data Uji'], y=[train_size, test_size],
                            text=[train_size, test_size], textposition='auto',
                            marker_color=['#66B3FF', '#FF9999'], width=0.8)])
    fig.update_layout(title="Perbandingan Data Training dan Uji",
                      yaxis_title="Jumlah Data", title_x=0.5, height=400)
    return fig

def plot_accuracy_comparison(results):
    fig = go.Figure([go.Bar(x=list(results.keys()), y=[results[name]['acc'] for name in results],
                            text=[f"{results[name]['acc']:.2f}" for name in results],
                            textposition='auto', marker_color=['#FF6B6B', '#4ECDC4'], width=0.8)])
    fig.update_layout(title="Perbandingan Akurasi Model", yaxis_title="Akurasi",
                      yaxis_range=[0, 1], title_x=0.5, height=400)
    return fig
