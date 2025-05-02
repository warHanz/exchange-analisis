import streamlit as st
import pandas as pd
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google_play_scraper import app, reviews
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Load CSS from external file
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found in 'assets/style.css'. Using default appearance.")


# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi utilitas

def preprocessing_data():
    st.write("Ini adalah proses preprocessing data.")

def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except:
        st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
        return None

def preprocess_data(data, required_columns=['Date', 'Username', 'Rating', 'Reviews Text']):
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing_columns)}")
        return None
    return data[required_columns].drop_duplicates(subset='Reviews Text', keep='first')

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    return re.sub(r'[^a-z\s]', '', text).strip()

def normalize_text(text, norm_dict):
    return ' '.join(norm_dict.get(word, word) for word in text.split())

def tokenize_text(text):
    return word_tokenize(text) if text else []

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    return [token for token in tokens if token not in stop_words]

def stem_tokens(tokens):
    return [stemmer.stem(token) for token in tokens]

def generate_wordcloud(tokens_list):
    # Gabungkan semua token menjadi satu string
    all_words = ' '.join([word for tokens in tokens_list for word in tokens])
    # Buat wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(all_words)
    return wordcloud

def generate_frequency_chart(tokens_list, top_n=10):
    all_words = [word for tokens in tokens_list for word in tokens]
    word_freq = Counter(all_words).most_common(top_n)
    words, frequencies = zip(*word_freq)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(words, frequencies)
    ax.set_title(f"Top {top_n} Kata Paling Sering Muncul")
    ax.set_xlabel("Kata")
    ax.set_ylabel("Frekuensi")
    plt.xticks(rotation=45)
    return fig

# Aplikasi Streamlit
def PreprocessingData():
    st.title("Sentiment Analysis")
    
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel", type=["csv", "xlsx"])
    if not uploaded_file:
        return

    data = load_data(uploaded_file)
    if data is None:
        return

    st.write("### Data Preview", data)
    st.write("### Informasi Data", data.describe(include='all'))

    # Step 1: Preprocessing
    st.subheader("Step 1: Preprocessing Data")
    if st.button("Preprocessing Data"):
        processed_data = preprocess_data(data)
        if processed_data is not None:
            st.session_state['preprocessed_data'] = processed_data
            st.session_state['step1_done'] = True

    if 'step1_done' in st.session_state and st.session_state['step1_done']:
        st.success(f"✔ {len(st.session_state['preprocessed_data'])} data unik setelah preprocessing")
        st.write("### Data Setelah Preprocessing", st.session_state['preprocessed_data'])
        st.write("### Informasi Data", st.session_state['preprocessed_data'].describe(include='all'))

    # Step 2: Cleaning
    if 'step1_done' in st.session_state and st.session_state['step1_done']:
        st.subheader("Step 2: Cleaning Data")
        if st.button("Cleaning Data"):
            data = st.session_state['preprocessed_data'].copy()
            data['Cleaned Reviews Text'] = data['Reviews Text'].apply(clean_text)
            data = data[data['Cleaned Reviews Text'] != '']
            data = data.dropna(subset=['Cleaned Reviews Text'])
            
            st.session_state['cleaned_data'] = data
            st.session_state['step2_done'] = True

    if 'step2_done' in st.session_state and st.session_state['step2_done']:
        st.success(f"✔ {len(st.session_state['cleaned_data'])} data setelah cleaning")
        st.write("### Data Setelah Cleaning", st.session_state['cleaned_data'])
        st.write("### Informasi Data", st.session_state['cleaned_data'].describe(include='all'))

    # Step 3: Normalisasi
    if 'step2_done' in st.session_state and st.session_state['step2_done']:
        st.subheader("Step 3: Normalisasi Data")
        if st.button("Normalisasi Data"):
            data = st.session_state['cleaned_data'].copy()
            try:
                kamus_kata_baku_path = os.path.join("assets", "kamuskatabaku.xlsx")
                kamus_kata_baku = pd.read_excel(kamus_kata_baku_path)
                norm_dict = dict(zip(kamus_kata_baku['tidak_baku'], kamus_kata_baku['kata_baku']))
                
                data['Normalized Reviews Text'] = data['Cleaned Reviews Text'].apply(
                    lambda x: normalize_text(x, norm_dict))
                
                st.session_state['normalized_data'] = data
                st.session_state['step3_done'] = True
                
            except Exception as e:
                st.error(f"Gagal memuat kamus: {e}")

    if 'step3_done' in st.session_state and st.session_state['step3_done']:
        st.success("✔ Normalisasi selesai")
        st.write("### Data Setelah Normalisasi", st.session_state['normalized_data'])

    # Step 4: Tokenizing
    if 'step3_done' in st.session_state and st.session_state['step3_done']:
        st.subheader("Step 4: Tokenizing Data")
        if st.button("Tokenizing Data"):
            data = st.session_state['normalized_data'].copy()
            data['Tokenized Reviews Text'] = data['Normalized Reviews Text'].apply(tokenize_text)
            
            st.session_state['tokenized_data'] = data
            st.session_state['step4_done'] = True

    if 'step4_done' in st.session_state and st.session_state['step4_done']:
        st.success("✔ Tokenisasi selesai")
        st.write("### Data Setelah Tokenisasi", st.session_state['tokenized_data'])
        st.write("### Informasi Data", st.session_state['tokenized_data'].describe(include='all'))

    # Step 5: Stopword Removal
    if 'step4_done' in st.session_state and st.session_state['step4_done']:
        st.subheader("Step 5: Remove Stopwords")
        if st.button("Remove Stopwords"):
            data = st.session_state['tokenized_data'].copy()
            data['Filtered Reviews Text'] = data['Tokenized Reviews Text'].apply(remove_stopwords)
            data = data[data['Filtered Reviews Text'].apply(len) > 0]
            
            st.session_state['filtered_data'] = data
            st.session_state['step5_done'] = True

    if 'step5_done' in st.session_state and st.session_state['step5_done']:
        st.success("✔ Stopword removal selesai")
        st.write("### Data Setelah Stopword Removal", st.session_state['filtered_data'])
        st.write("### Informasi Data", st.session_state['filtered_data'].describe(include='all'))

    # Step 6: Stemming
    if 'step5_done' in st.session_state and st.session_state['step5_done']:
        st.subheader("Step 6: Stemming Data")
        if st.button("Stemming Data"):
            data = st.session_state['filtered_data'].copy()
            data['Stemmed Reviews Text'] = data['Filtered Reviews Text'].apply(stem_tokens)
            
            st.session_state['stemmed_data'] = data
            st.session_state['step6_done'] = True

    if 'step6_done' in st.session_state and st.session_state['step6_done']:
        st.success("✔ Stemming selesai")
        st.write("### Data Setelah Stemming", st.session_state['stemmed_data'])
        st.write("### Informasi Data", st.session_state['stemmed_data'].describe(include='all'))

      # Step 7: Wordcloud
    if 'step6_done' in st.session_state and st.session_state['step6_done']:
        st.subheader("Step 7: Generate Wordcloud")
        if st.button("Generate Wordcloud"):
            data = st.session_state['stemmed_data'].copy()
            wordcloud = generate_wordcloud(data['Stemmed Reviews Text'])
            
            st.session_state['wordcloud'] = wordcloud
            st.session_state['step7_done'] = True

    if 'step7_done' in st.session_state and st.session_state['step7_done']:
        st.success("✔ Wordcloud berhasil dibuat")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(st.session_state['wordcloud'], interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

     # Step 8: Frequency Chart
    if 'step7_done' in st.session_state and st.session_state['step7_done']:
        st.subheader("Step 8: Generate Frequency Chart")
        if st.button("Generate Frequency Chart"):
            data = st.session_state['stemmed_data'].copy()
            freq_chart = generate_frequency_chart(data['Stemmed Reviews Text'], top_n=10)
            
            st.session_state['freq_chart'] = freq_chart
            st.session_state['step8_done'] = True

    if 'step8_done' in st.session_state and st.session_state['step8_done']:
        st.success("✔ Frequency chart berhasil dibuat")
        st.pyplot(st.session_state['freq_chart'])

if __name__ == "__main__":
    PreprocessingData()