import streamlit as st
from PIL import Image

def developer():
    # Konfigurasi halaman
    st.set_page_config(page_title="Tim Pengembang", layout="wide")

    # Custom CSS for card effect and centering
    st.markdown("""
        <style>
        .dev-card {
            background: #f8f9fa;
            border-radius: 18px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
            padding: 32px 16px 16px 16px;
            margin: 16px 0;
            text-align: center;
            transition: box-shadow 0.2s;
        }
        .dev-card:hover {
            box-shadow: 0 8px 32px rgba(0,0,0,0.16);
        }
        .dev-img {
            border-radius: 50%;
            border: 4px solid #e0e0e0;
            margin-bottom: 16px;
            object-fit: cover;
        }
        .dev-name {
            font-size: 1.25rem;
            font-weight: 600;
            margin-top: 12px;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 style='text-align: center; margin-bottom: 10;'>👨‍💻 Tim Pengembang</h1>", unsafe_allow_html=True)
    # st.markdown("<p style='text-align: center; color: #666; margin-top: 0;'>Berikut adalah tim pengembang dari sistem ini:</p>", unsafe_allow_html=True)

    # Data pengembang
    developers = [
        {"name": "Reihan Nanda Muliawan", "image": "assets/pengembang/reihan.jpg"},
        {"name": "Jemmy Edwin Bororing", "image": "assets/pengembang/pakjemmy.jpeg"},
        {"name": "Yumarlin M.Z", "image": "assets/pengembang/bumarlin.jpeg"}
    ]

    # Tampilkan dalam 3 kolom di tengah
    cols = st.columns([1, 1, 1], gap="large")

    for col, dev in zip(cols, developers):
        with col:
            try:
                image = Image.open(dev["image"]).convert("RGB")
                image = image.resize((200, 200))
                st.markdown(
                    f"""
                    <div class="dev-card">
                        <img src="data:image/jpeg;base64,{image_to_base64(image)}" class="dev-img" width="160" height="160"/>
                        <div class="dev-name">{dev['name']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except FileNotFoundError:
                st.error(f"Gambar {dev['image']} tidak ditemukan.")

def image_to_base64(image):
    import io, base64
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return base64.b64encode(byte_im).decode()

if __name__ == "__main__":
    developer()
