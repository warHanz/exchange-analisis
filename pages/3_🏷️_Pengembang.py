import streamlit as st
from PIL import Image

def developer():
    # Konfigurasi halaman
    st.set_page_config(page_title="Tim Pengembang", layout="wide")

    # Header
    st.markdown("<h1 style='text-align: center;'>👨‍💻 Tim Pengembang</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Berikut adalah tim pengembang dari sistem ini:</p>", unsafe_allow_html=True)

    # Data pengembang
    developers = [
        {"name": "Reihan", "image": "assets/pengembang/reihan.jpg"},
        {"name": "Fahmi", "image": "assets/pengembang/reihan.jpg"},
        {"name": "Aulia", "image": "assets/pengembang/reihan.jpg"}
    ]

    # Tampilkan dalam 3 kolom di tengah
    cols = st.columns([1, 1, 1], gap="large")

    # Perbesar ukuran gambar
    for col, dev in zip(cols, developers):
        with col:
            try:
                image = Image.open(dev["image"])
                st.image(image, width=250, caption=dev["name"])  # Ukuran gambar diperbesar
            except FileNotFoundError:
                st.error(f"Gambar {dev['image']} tidak ditemukan.")


if __name__ == "__main__":
    developer()
