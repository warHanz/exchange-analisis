import streamlit as st
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
import nltk

# Unduh tokenizer NLTK yang diperlukan
nltk.download('punkt')

# Memuat CSS dari file eksternal
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.warning("File CSS tidak ditemukan di 'assets/style.css'. Menggunakan tampilan default.")

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


# Fungsi utilitas
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    except:
        st.error("Format file tidak didukung. Harap unggah file CSV atau Excel.")
        return None

def preprocess_for_model(data):
    def safe_convert(x):
        try:
            if isinstance(x, str):
                tokens = ast.literal_eval(x)
                return ' '.join(tokens)
            elif isinstance(x, list):
                return ' '.join(x)
            return str(x)
        except (ValueError, SyntaxError):
            return str(x)
    data['Text_for_Model'] = data['Stemmed Reviews Text'].apply(safe_convert)
    return data

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    return fig

def plot_sentiment_bar(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=data, x='Sentiment', order=['Negative', 'Neutral', 'Positive'], ax=ax)
    ax.set_title('Distribusi Sentimen (Bar Chart)')
    ax.set_xlabel('Sentimen')
    ax.set_ylabel('Jumlah')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='baseline', xytext=(0, 5), textcoords='offset points')
    return fig

def plot_sentiment_pie(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sentiment_counts = data['Sentiment'].value_counts()
    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    ax.set_title('Distribusi Sentimen (Pie Chart)')
    ax.axis('equal')
    return fig

def plot_data_split(train_size, test_size):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=['Data Training', 'Data Uji'], y=[train_size, test_size], ax=ax, palette='Blues')
    ax.set_title('Perbandingan Jumlah Data Training dan Uji')
    ax.set_ylabel('Jumlah Data')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='baseline', xytext=(0, 5), textcoords='offset points')
    return fig

def plot_model_accuracy(svm_acc, nb_acc):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=['SVM', 'Naive Bayes'], y=[svm_acc, nb_acc], ax=ax, palette='Greens')
    ax.set_title('Perbandingan Akurasi Model')
    ax.set_ylabel('Akurasi')
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='baseline', xytext=(0, 5), textcoords='offset points')
    return fig

def plot_wordcloud(text, title, color):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=color, max_words=100).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    return fig

# Aplikasi Streamlit
def main():
    st.title("Sentiment Analysis - Model Training and Evaluation")
    
    uploaded_file = st.file_uploader("Unggah file CSV atau Excel hasil stemming", type=["csv", "xlsx"])
    if not uploaded_file:
        return

    data = load_data(uploaded_file)
    if data is None:
        return

    st.write("### Data Preview", data.head())
    st.write("### Informasi Data", data.describe(include='all'))

    required_columns = ['Stemmed Reviews Text', 'Rating']
    if not all(col in data.columns for col in required_columns):
        st.error("Data harus memiliki kolom 'Stemmed Reviews Text' dan 'Rating'")
        return

    # Step 1: Pelabelan Data
    st.subheader("Step 1: Pelabelan Data")
    if st.button("Label Data"):
        positive_lexicon, negative_lexicon = load_sentiment_lexicon()
        
        def label_sentiment(text):
            tokens = word_tokenize(text)
            positive_count = sum(1 for token in tokens if token in positive_lexicon)
            negative_count = sum(1 for token in tokens if token in negative_lexicon)
            sentiment_score = positive_count - negative_count
    
            if sentiment_score > 0:
                sentiment = "Positive"
            elif sentiment_score < 0:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
    
            return sentiment
        
        data['Sentiment'] = data['Stemmed Reviews Text'].apply(label_sentiment)
        data = preprocess_for_model(data)
        
        st.session_state['labeled_data'] = data
        st.session_state['step1_done'] = True

    if 'step1_done' in st.session_state and st.session_state['step1_done']:
        st.success("✔ Pelabelan data selesai (Positif/Netral/Negatif)")
        st.write("### Hasil Pelabelan", st.session_state['labeled_data'][['Stemmed Reviews Text', 'Text_for_Model', 'Sentiment']])
        st.write("### Distribusi Sentimen")
        st.pyplot(plot_sentiment_bar(st.session_state['labeled_data']))
        st.pyplot(plot_sentiment_pie(st.session_state['labeled_data']))

        st.write("### Word Cloud Berdasarkan Sentimen")
        labeled_data = st.session_state['labeled_data']
        positive_text = ' '.join(labeled_data[labeled_data['Sentiment'] == 'Positive']['Text_for_Model'])
        neutral_text = ' '.join(labeled_data[labeled_data['Sentiment'] == 'Neutral']['Text_for_Model'])
        negative_text = ' '.join(labeled_data[labeled_data['Sentiment'] == 'Negative']['Text_for_Model'])

        if positive_text:
            st.pyplot(plot_wordcloud(positive_text, "Word Cloud - Sentimen Positif", "Greens"))
        if neutral_text:
            st.pyplot(plot_wordcloud(neutral_text, "Word Cloud - Sentimen Netral", "Blues"))
        if negative_text:
            st.pyplot(plot_wordcloud(negative_text, "Word Cloud - Sentimen Negatif", "Reds"))

    # Step 2: Splitting dan Training
    if 'step1_done' in st.session_state and st.session_state['step1_done']:
        st.subheader("Step 2: Training Model")
        if st.button("Train Models"):
            data = st.session_state['labeled_data'].copy()
            
            X = data['Text_for_Model']
            y = data['Sentiment']
            X_train, X_test, y_train, y_test = split_data(X, y)
            
            vectorizer = TfidfVectorizer()
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            svm_model = SVC(kernel='linear')
            svm_model.fit(X_train_tfidf, y_train)
            svm_predictions = svm_model.predict(X_test_tfidf)

            nb_model = MultinomialNB()
            nb_model.fit(X_train_tfidf, y_train)
            nb_predictions = nb_model.predict(X_test_tfidf)

            svm_accuracy = accuracy_score(y_test, svm_predictions)
            nb_accuracy = accuracy_score(y_test, nb_predictions)

            st.session_state['X_train_shape'] = X_train.shape[0]
            st.session_state['X_test_tfidf'] = X_test_tfidf
            st.session_state['y_test'] = y_test
            st.session_state['svm_pred'] = svm_predictions
            st.session_state['nb_pred'] = nb_predictions
            st.session_state['svm_accuracy'] = svm_accuracy
            st.session_state['nb_accuracy'] = nb_accuracy
            st.session_state['step2_done'] = True

    if 'step2_done' in st.session_state and st.session_state['step2_done']:
        st.success("✔ Model SVM dan Naive Bayes telah dilatih")
        st.write(f"Jumlah data training: {st.session_state['X_train_shape']} | Jumlah data uji: {st.session_state['X_test_tfidf'].shape[0]}")
        st.write("### Visualisasi Hasil Training")
        st.pyplot(plot_data_split(st.session_state['X_train_shape'], st.session_state['X_test_tfidf'].shape[0]))

    # Step 3: Evaluasi Model
    if 'step2_done' in st.session_state and st.session_state['step2_done']:
        st.subheader("Step 3: Evaluasi Model")
        if st.button("Evaluate Models"):
            y_test = st.session_state['y_test']
            svm_predictions = st.session_state['svm_pred']
            nb_predictions = st.session_state['nb_pred']

            svm_cm = confusion_matrix(y_test, svm_predictions, labels=['Negative', 'Neutral', 'Positive'])
            svm_report = classification_report(y_test, svm_predictions)

            nb_cm = confusion_matrix(y_test, nb_predictions, labels=['Negative', 'Neutral', 'Positive'])
            nb_report = classification_report(y_test, nb_predictions)

            st.session_state['svm_cm'] = svm_cm
            st.session_state['svm_report'] = svm_report
            st.session_state['nb_cm'] = nb_cm
            st.session_state['nb_report'] = nb_report
            st.session_state['step3_done'] = True

    if 'step3_done' in st.session_state and st.session_state['step3_done']:
        st.write("## Hasil Evaluasi SVM")
        st.write(f"Akurasi: {st.session_state['svm_accuracy']:.2f}")
        st.pyplot(plot_confusion_matrix(st.session_state['svm_cm'], "Confusion Matrix - SVM (Negatif/Netral/Positif)"))
        st.text("Laporan Klasifikasi SVM:\n" + st.session_state['svm_report'])

        st.write("## Hasil Evaluasi Naive Bayes")
        st.write(f"Akurasi: {st.session_state['nb_accuracy']:.2f}")
        st.pyplot(plot_confusion_matrix(st.session_state['nb_cm'], "Confusion Matrix - Naive Bayes (Negatif/Netral/Positif)"))
        st.text("Laporan Klasifikasi Naive Bayes:\n" + st.session_state['nb_report'])

        # Grafik perbandingan akurasi
        st.write("### Perbandingan Akurasi Model")
        st.pyplot(plot_model_accuracy(st.session_state['svm_accuracy'], st.session_state['nb_accuracy']))

if __name__ == "__main__":
    main()