import streamlit as st

def guide():
    st.set_page_config(
        page_title="Panduan Pengguna - Analisis Sentimen",
        layout="wide"
    )

    st.title("🚀 Panduan Pengguna Aplikasi Analisis Sentimen")

    st.markdown(
        """
        Selamat datang di aplikasi Analisis Sentimen.  
        Panduan ini menunjukkan langkah-langkah utama proses mulai dari pengambilan data hingga visualisasi hasil.  
        Perhatikan diagram sebelah kiri dan deskripsi langkah prosedur di sebelah kanan.
        """
    )

    col1, col2 = st.columns([1, 1])

    graphviz_code = """
    digraph {
        rankdir=TB;  // Top to Bottom (default)
        node [shape=box, style=filled, color="#FF4B4B", fontname="Arial", fontsize=14, fontcolor=white, width=2.8, height=0.8];
        edge [color="#FF4B4B", penwidth=2];

        Scraping [label="Scraping Data"];
        Preprocessing [label="Preprocessing Data"];
        SentimentAnalysis [label="Analisis Sentimen"];
        Visualization [label="Visualisasi Hasil"];

        Scraping -> Preprocessing;
        Preprocessing -> SentimentAnalysis;
        SentimentAnalysis -> Visualization;
    }
    """

    with col1:
        st.subheader("Alur Proses")
        st.graphviz_chart(graphviz_code, use_container_width=True)

    with col2:
        st.subheader("Penjelasan Prosedur")

        st.markdown("### 1. Scraping Data")
        st.write(
            """
            - Mengumpulkan data dari sumber eksternal seperti media sosial, forum, atau situs ulasan.
            - Mengambil data dalam bentuk teks yang relevan untuk dianalisis.
            - Memastikan data bersih dari duplikasi dan sesuai kebutuhan analisis.
            """
        )

        st.markdown("### 2. Preprocessing Data")
        st.write(
            """
            - Membersihkan teks dari karakter spesial, tautan, dan tanda baca yang tidak diperlukan.
            - Melakukan tokenisasi, normalisasi, dan menghapus kata tidak penting (stopwords).
            - Menyiapkan data agar siap dianalisis dengan hasil lebih akurat.
            """
        )

        st.markdown("### 3. Analisis Sentimen")
        st.write(
            """
            - Menerapkan model untuk menentukan sentimen teks: positif, negatif, atau netral.
            - Model dapat menggunakan metode machine learning atau lexicon-based.
            - Menghasilkan label sentimen dan skor kepercayaan untuk setiap data.
            """
        )

        st.markdown("### 4. Visualisasi Hasil")
        st.write(
            """
            - Menampilkan hasil analisis dalam bentuk grafik atau dashboard interaktif.
            - Memudahkan pemahaman distribusi sentimen dan insight penting.
            - Membantu pengambilan keputusan berbasis hasil analisis.
            """
        )

    st.markdown("---")
    st.info("📌 Gunakan menu utama aplikasi untuk mulai proses analisis sentimen Anda.")

if __name__ == "__main__":
    guide()
