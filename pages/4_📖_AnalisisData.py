import streamlit as st
from utils.utils import *
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurasi Streamlit
st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")

def main():
    st.title("📊 Sentiment Analysis Dashboard")
    st.markdown("Upload file CSV/Excel untuk analisis sentimen otomatis")

    uploaded_file = st.file_uploader("Unggah file dataset (CSV/Excel)", type=["csv", "xlsx"], key="data_uploader")
    
    if uploaded_file is None:
        st.info("Silakan unggah file dataset untuk memulai analisis.")
        return

    with st.spinner("Memproses data..."):
        # Memuat leksikon sentimen dari URL
        positive_words, negative_words = load_sentiment_lexicon()

        data = load_data(uploaded_file)
        if data is None:
            return

        # 1. Data Overview - Sebelum Diproses
        st.subheader("📋 Data Overview - Sebelum Diproses")
        st.write("Ringkasan dataset asli yang diunggah.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Total Data**")
            st.write(f"{len(data)} Ulasan")
        with col2:
            avg_rating = data['Rating'].mean().round(2) if 'Rating' in data.columns else "N/A"
            st.write("**Rata-rata Rating**")
            st.write(avg_rating)
        
        if 'Date' in data.columns:
            try:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data['Month-Year'] = data['Date'].dt.to_period('M').astype(str)
                monthly_counts = data.groupby('Month-Year').size().reset_index(name='Jumlah Ulasan')
                fig = px.bar(monthly_counts, x='Month-Year', y='Jumlah Ulasan', 
                            title="Jumlah Ulasan per Bulan",
                            color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Gagal memproses kolom tanggal: {e}")
        else:
            st.warning("Kolom 'Date' tidak ditemukan. Grafik per bulan tidak dapat ditampilkan.")

        # Proses preprocessing
        processed_data = preprocess_data(data)
        if processed_data is None:
            return
        
        processed_data['Processing'] = processed_data['Reviews Text']
        processed_data['Clean'] = processed_data['Processing'].apply(clean_text)
        norm_dict = None
        kamus_path = os.path.join("assets", "kamuskatabaku.xlsx")
        if os.path.exists(kamus_path):
            try:
                kamus = pd.read_excel(kamus_path)
                norm_dict = dict(zip(kamus['tidak_baku'], kamus['kata_baku']))
                processed_data['Normalize'] = processed_data['Clean'].apply(lambda x: normalize_text(x, norm_dict))
            except Exception as e:
                st.warning(f"Gagal membaca kamuskatabaku.xlsx: {e}")
                processed_data['Normalize'] = processed_data['Clean']
        else:
            st.warning("File 'kamuskatabaku.xlsx' tidak ditemukan. Normalisasi dilewati.")
            processed_data['Normalize'] = processed_data['Clean']
        
        processed_data['Tokenize'] = processed_data['Normalize'].apply(tokenize_text)
        processed_data['Remove Stopword'] = processed_data['Tokenize'].apply(remove_stopwords)
        processed_data['Stemmed'] = processed_data['Remove Stopword'].apply(stem_tokens)
        
        processed_data[['Sentiment Score', 'Sentiment']] = processed_data['Stemmed'].apply(
            lambda x: pd.Series(label_sentiment(x, positive_words, negative_words)))
        processed_data['Text_for_Model'] = processed_data['Stemmed'].apply(preprocess_for_model)
        processed_data['Word Count'] = processed_data['Stemmed'].apply(len)

        processed_data = processed_data[processed_data['Stemmed'].apply(len) > 0]

        # Log distribusi kelas
        logger.info(f"Distribusi kelas: {processed_data['Sentiment'].value_counts().to_dict()}")

        # 2. Data Overview - Setelah Diproses
        st.subheader("📋 Data Overview - Setelah Diproses")
        st.write("Ringkasan data setelah preprocessing.")
        
        fig = px.histogram(processed_data, x='Word Count', nbins=30, 
                          title="Distribusi Jumlah Kata Setelah Stemming",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("Tabel: Teks asli, dibersihkan, dinormalisasi, ditokenisasi, dan stopwords dihapus.")
        display_data = processed_data[['Processing', 'Clean', 'Normalize', 'Tokenize', 'Remove Stopword', 'Stemmed']].copy()
        display_data.index = display_data.index + 1
        st.dataframe(display_data)

        # 3. Sentimen
        st.subheader("📋 Sentimen")
        st.write("Tabel ini menampilkan skor sentimen dan label sentimen berdasarkan leksikon.")
        sentiment_data = processed_data[['Stemmed', 'Sentiment Score', 'Sentiment']].copy()
        sentiment_data.index = sentiment_data.index + 1
        st.dataframe(sentiment_data)
        st.write("Distribusi Sentimen:", processed_data['Sentiment'].value_counts())

        # Pengecekan jumlah kelas unik
        unique_classes = processed_data['Sentiment'].nunique()
        if unique_classes <= 1:
            st.error(f"Hanya ditemukan {unique_classes} kelas sentimen: {processed_data['Sentiment'].unique()}. "
                     "Model tidak dapat dilatih karena memerlukan minimal 2 kelas.")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_data[['Text_for_Model', 'Word Count', 'Sentiment Score']], 
            processed_data['Sentiment'], test_size=0.2, random_state=42, stratify=processed_data['Sentiment'])

        # Vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2, max_df=0.8, sublinear_tf=True)
        X_train_tfidf = vectorizer.fit_transform(X_train['Text_for_Model'])
        X_test_tfidf = vectorizer.transform(X_test['Text_for_Model'])

        # Tambahkan fitur tambahan untuk SVM
        X_train_full = create_features(X_train, vectorizer)
        X_test_full = create_features(X_test, vectorizer)

        # Penanganan ketidakseimbangan kelas dengan SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_full, y_train)
        # SMOTE untuk Naive Bayes (hanya TF-IDF)
        X_train_tfidf_balanced, y_train_tfidf_balanced = smote.fit_resample(X_train_tfidf, y_train)

        # Definisikan model dan hyperparameter untuk tuning
        models = {
            'SVM': GridSearchCV(
                SVC(probability=True, class_weight='balanced'),
                param_grid={'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
                cv=5, scoring='accuracy', n_jobs=-1
            ),
            'Naive Bayes': GridSearchCV(
                MultinomialNB(),
                param_grid={'alpha': [0.1, 0.5, 1.0]},
                cv=5, scoring='accuracy', n_jobs=-1
            )
        }

        results = {}
        for name, model in models.items():
            if name == 'Naive Bayes':
                # Gunakan hanya TF-IDF untuk Naive Bayes
                model.fit(X_train_tfidf_balanced, y_train_tfidf_balanced)
                predictions = model.predict(X_test_tfidf)
                cv_mean, cv_std = evaluate_model(model.best_estimator_, X_train_tfidf_balanced, y_train_tfidf_balanced)
            else:
                # Gunakan fitur lengkap untuk SVM
                model.fit(X_train_balanced, y_train_balanced)
                predictions = model.predict(X_test_full)
                cv_mean, cv_std = evaluate_model(model.best_estimator_, X_train_balanced, y_train_balanced)
            
            results[name] = {
                'pred': predictions,
                'acc': accuracy_score(y_test, predictions),
                'report': classification_report(y_test, predictions, output_dict=True),
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'best_params': model.best_params_ if name in ['SVM', 'Naive Bayes'] else None
            }

    # Visualisasi Preprocessing
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌐 Visualisasi Preprocessing")
        tabs = st.tabs(["Semua Data", "Positif", "Netral", "Negatif"])
        
        with tabs[0]:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(generate_wordcloud(processed_data['Stemmed'], "Semua Data", 'viridis'), interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        for tab, sentiment, color in zip(tabs[1:], ['Positive', 'Neutral', 'Negative'], 
                                        ['Greens', 'Greys', 'Reds']):
            with tab:
                sentiment_data = processed_data[processed_data['Sentiment'] == sentiment]['Stemmed']
                if len(sentiment_data) > 0:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.imshow(generate_wordcloud(sentiment_data, f"{sentiment}", color), interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)

    with col2:
        st.subheader("📈 Distribusi Sentimen")
        sentiment_counts = processed_data['Sentiment'].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                    title="Distribusi Sentimen", hole=0.3,
                    color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    # Top 10 Kata
    st.subheader("📊 Top 10 Kata Paling Sering Muncul")
    st.plotly_chart(generate_frequency_chart(processed_data['Stemmed']), use_container_width=True)

    # Hasil Model
    st.subheader("🚀 Hasil Model")
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.plotly_chart(plot_data_split(len(X_train), len(X_test)), use_container_width=True)
    
    with col_model2:
        st.plotly_chart(plot_accuracy_comparison(results), use_container_width=True)

    for name in models:
        st.write(f"### {name}")
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            st.plotly_chart(plot_confusion_matrix(y_test, results[name]['pred'], name),
                          use_container_width=True)
            st.write(f"**Cross-Validation Accuracy**: {results[name]['cv_mean']:.2f} ± {results[name]['cv_std']:.2f}")
            if results[name]['best_params']:
                st.write(f"**Best Parameters**: {results[name]['best_params']}")
        with col_eval2:
            report_df = pd.DataFrame(results[name]['report']).T.round(2)
            st.write("Laporan Klasifikasi:")
            report_df.index.name = 'Class'
            report_df = report_df.reset_index()
            st.dataframe(report_df)

if __name__ == "__main__":
    main()