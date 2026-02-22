import streamlit as st
from PIL import Image, ImageChops
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

# --- Load Model and Dataset ---
model = tf.keras.models.load_model("fashion_category_model.h5")
df = pd.read_pickle("df_clear.pkl")
class_labels = ['Topwear', 'Bottomwear', 'OnePiece', 'Footwear', 'Accessories']

# --- Utility Functions ---
def auto_crop_image(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img

def extract_dominant_color_clean(img, k=3):
    img = auto_crop_image(img.convert('RGB')).resize((100, 100))
    img_np = np.array(img)
    pixels = img_np.reshape(-1, 3)
    pixels = np.array([p for p in pixels if not (
        p[0] > 240 and p[1] > 240 and p[2] > 240
    )])
    if len(pixels) == 0:
        return (200, 200, 200)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    return tuple(kmeans.cluster_centers_[0].astype(int))

def color_distance(c1, c2):
    return euclidean_distances([c1], [c2])[0][0]

category_flow = {
    'Topwear': ['Bottomwear', 'Footwear', 'Accessories'],
    'Bottomwear': ['Topwear', 'Footwear', 'Accessories'],
    'OnePiece': ['Footwear', 'Accessories']
}

def recommend_outfit(img_pil, gender):
    resized = img_pil.convert('RGB').resize((224, 224))
    img_array = keras_image.img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    pred_label = class_labels[pred_index]

    dom_color = np.array(extract_dominant_color_clean(img_pil))
    recommendations = {}

    if pred_label not in category_flow:
        return pred_label, recommendations

    for cat in category_flow[pred_label]:
        subset = df[(df['mapped_category'] == cat) & (df['gender'].str.lower() == gender.lower())].copy()
        if subset.empty:
            continue
        subset['color_distance'] = subset['dominant_color'].apply(
            lambda x: color_distance(dom_color, np.array(x))
        )
        top_match = subset.sort_values(by='color_distance').iloc[0]
        recommendations[cat] = top_match['image_path']

    return pred_label, recommendations

# --- Modernized CSS + Layout ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(to bottom, #fef6f9, #eef2f7);
        }
        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
        }
        .main-title {
            font-size: 2.6rem;
            font-weight: 600;
            text-align: center;
            background: linear-gradient(to right, #ff416c, #ff4b2b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }
        .subheading {
            font-size: 1.1rem;
            text-align: center;
            color: #555;
            margin-bottom: 2.5rem;
        }
        .info-box {
            background-color: #e7f3fe;
            border-left: 5px solid #2196F3;
            padding: 12px 15px;
            border-radius: 10px;
            font-size: 1.05rem;
            color: #0d47a1;
            margin: 20px 0 30px 0;
        }
        .image-card {
            padding: 12px;
            background: #ffffffcc;
            border-radius: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            text-align: center;
            transition: all 0.3s ease;
        }
        .image-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        .image-card img {
            border-radius: 12px;
            margin-bottom: 8px;
        }
        .footer {
            text-align: center;
            font-size: 0.85rem;
            margin-top: 3rem;
            color: #888;
        }
        .stRadio > div {
            flex-direction: row;
            justify-content: center;
            gap: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- App UI ---
st.markdown("<div class='main-title'>An AI-Powered Personalized Clothing Recommendation Using Deep Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='subheading'>Upload a clothing item and receive full outfit suggestions with smart color coordination.</div>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload Clothing Image", type=["jpg", "jpeg", "png"])
gender = st.radio("Select Gender", ["Women", "Men"])

if uploaded_image:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", width=250)

    with st.spinner("Analyzing your style..."):
        predicted_label, recs = recommend_outfit(img, gender)

    st.markdown(f"<div class='info-box'>Predicted Category: <strong>{predicted_label}</strong></div>", unsafe_allow_html=True)

    if not recs:
        st.warning("No matching outfit items found for the selected gender.")
    else:
        st.markdown("### Recommended Outfit Items")
        cols = st.columns(len(recs))
        for i, (cat, path) in enumerate(recs.items()):
            with cols[i]:
                st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                st.image(path, width=180, caption=cat)
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>© 2025 · StyleSnap: Style Recommendation System</div>", unsafe_allow_html=True)