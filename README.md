# 👗 AI-Powered Personalized Clothing Recommendation System

## 📌 Overview

This project is an AI-driven fashion recommendation system that analyzes a user-uploaded clothing image and suggests matching outfit items. The system classifies clothing into categories such as topwear, bottomwear, footwear, and accessories, considers user gender, extracts dominant color features, and recommends complementary items using machine learning techniques.

The goal is to demonstrate how deep learning and data-driven logic can be applied to build intelligent fashion recommendation systems that enhance user experience.

---

## ✨ Features

* Image-based clothing classification using a trained deep learning model
* Gender selection for personalized recommendations
* Dominant color extraction using clustering (KMeans)
* Outfit recommendation logic based on category relationships
* Interactive user interface built with Streamlit
* Real-time prediction and recommendation display
* Clean and modern UI for user interaction

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras (Deep Learning)
* Streamlit (Web Interface)
* NumPy & Pandas (Data Processing)
* Scikit-learn (KMeans Clustering & Similarity)
* Pillow (Image Processing)

---

## 🧠 How It Works

1. User uploads a clothing image.
2. User selects gender (Men or Women).
3. The deep learning model predicts the clothing category.
4. Dominant color is extracted from the image.
5. System searches the processed dataset (`df_clear.pkl`) for matching items.
6. Recommendations are generated based on category flow and color similarity.
7. Results are displayed in the interface.

---

## 📂 Project Structure

```
.
├── app.py                         # Streamlit application
├── fashion_category_model.h5      # Trained classification model
├── df_clear.pkl                   # Processed dataset with features
├── requirements.txt               # Dependencies
├── screenshots/                   # App visuals
└── README.md
```

---

## 🚀 How to Run Locally

### 1️⃣ Clone repository

```
git clone <your-repo-link>
cd <project-folder>
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run application

```
streamlit run app.py
```

### 4️⃣ Open browser

```
http://localhost:8501
```

---

## 📊 Dataset

The project uses a fashion dataset derived from the Myntra product dataset, which includes clothing images and metadata stored in a CSV file.

Due to size limitations, the full dataset and image files are not included in this repository. The application relies on a processed dataset file (`df_clear.pkl`) for recommendations.

---

## 📸 Screenshots

![Home](screenshots/home.png)
![Upload](screenshots/upload.png,screenshots/uploads.png)
![Prediction](screenshots/prediction.png)
![Recommendations](screenshots/recommendation.png)

---

## ⚠️ Limitations

* The model may occasionally misclassify visually similar items due to overlapping features.
* Recommendation quality depends on dataset coverage.
* Performance may vary for unseen clothing styles.

---

## 🔮 Future Improvements

* Improve model accuracy with larger training data
* Add confidence score display
* use high accuracy dataset (good pixel quality)
* Implement advanced recommendation algorithms
* Deploy as a cloud-hosted web application
* Improve UI responsiveness
* Integrate user preference learning

---

## 📄 Publication

This project is associated with research published in IJSRED (International Journal of Scientific Research and Engineering Development).

---

## 🎯 Learning Outcomes

* Building end-to-end machine learning applications
* Integrating deep learning with user interfaces
* Working with real-world datasets
* Designing recommendation logic
* Handling model inference pipelines

---

## 🤝 Contribution

Developed as part of a team project focusing on system design, model integration, and implementation.

---

## 📜 License

This project is intended for academic and learning purposes.
