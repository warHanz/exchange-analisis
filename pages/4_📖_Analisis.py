import streamlit as st
from utils.utils import *
from models.models import prepare_data, train_and_evaluate_models

# Konfigurasi Streamlit
st.set_page_config(layout="wide", page_title="Sentiment Analysis Dashboard")

def main():
    st.title("Sentiment Analysis Dashboard")
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
        st.subheader("📋 Data Overview")
        st.write("Ringkasan dataset asli yang diunggah.")
        
        # Total Data dan Average Rating
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Total Data**")
            st.write(f"{len(data)} Ulasan")
        with col2:
            avg_rating = data['Rating'].mean().round(2) if 'Rating' in data.columns else "N/A"
            st.write("**Rata-rata Rating**")
            st.write(avg_rating)
        
        # Grafik Date per Bulan
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

        processed_data = processed_data[processed_data['Stemmed'].apply(len) > 0]

        # 2. Data Overview - Setelah Diproses
        st.subheader("📋 Tabel")
        
        st.write("Tabel: Teks asli, dibersihkan, dinormalisasi, ditokenisasi, dan stopwords dihapus.")
        st.dataframe(processed_data[['Processing', 'Clean', 'Normalize', 'Tokenize', 'Remove Stopword', 'Stemmed']])

        # 3. Sentimen
        st.subheader("📋 Sentimen")
        st.write("Tabel ini menampilkan skor sentimen dan label sentimen berdasarkan leksikon dari URL setelah proses stemming.")
        st.dataframe(processed_data[['Stemmed', 'Sentiment Score', 'Sentiment']])
        st.write("Distribusi Sentimen:", processed_data['Sentiment'].value_counts())

        # Pengecekan jumlah kelas unik
        unique_classes = processed_data['Sentiment'].nunique()
        if unique_classes <= 1:
            st.error(f"Hanya ditemukan {unique_classes} kelas sentimen: {processed_data['Sentiment'].unique()}. "
                     "Model tidak dapat dilatih karena memerlukan minimal 2 kelas.")
            return

        # Split data dan pelatihan model menggunakan models.py
        X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = prepare_data(
            processed_data['Text_for_Model'], processed_data['Sentiment']
        )
        results = train_and_evaluate_models(X_train_tfidf, X_test_tfidf, y_train, y_test)

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
        # Definisikan warna untuk masing-masing sentimen
        colors = {
            'Positive': '#00FF00',  # Hijau
            'Negative': '#FF0000',  # Merah
            'Neutral': '#808080'    # Abu-abu
        }
        # Urutkan warna sesuai dengan indeks sentimen
        color_list = [colors.get(sentiment, '#808080') for sentiment in sentiment_counts.index]
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribusi Sentimen",
            hole=0.3,
            color_discrete_sequence=color_list  # Gunakan warna yang ditentukan
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top 10 Kata
    st.subheader("📊 Top 10 Kata Paling Sering Muncul")
    st.plotly_chart(generate_frequency_chart(processed_data['Stemmed']), use_container_width=True)

    # Hasil Model
    st.subheader("🚀 Hasil Model")
    col_model1, col_model2 = st.columns(2)

    with col_model1:
        st.plotly_chart(plot_data_split(X_train_tfidf.shape[0], X_test_tfidf.shape[0]), use_container_width=True)

    with col_model2:
        st.plotly_chart(plot_accuracy_comparison(results), use_container_width=True)

    for name in results:
        st.write(f"### {name}")
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            st.plotly_chart(plot_confusion_matrix(y_test, results[name]['pred'], name),
                          use_container_width=True)
        with col_eval2:
            report_df = pd.DataFrame(results[name]['report']).T.round(2)
            st.write("Laporan Klasifikasi:")
            st.dataframe(report_df.style.highlight_max(axis=0, subset=['precision', 'recall', 'f1-score']))

if __name__ == "__main__":
    main()