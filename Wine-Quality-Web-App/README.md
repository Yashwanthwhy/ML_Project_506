# 🍷 Wine Quality Prediction Web Application

Live interactive Web App powered by a tuned **Random Forest Classifier** and **Flask**, deployed in real time on Vercel.

🔗 **Live Deployment:** [https://wine-quality-app-eight.vercel.app](https://wine-quality-app-eight.vercel.app)

---

## ✨ Features

- **Real-Time Wine Quality Classification:** Predicts whether a wine is **GOOD** (score $\ge 6$) or **POOR** with probability percentage and model confidence.
- **Dynamic Physicochemical Inputs:** 11 sliders for chemical attributes (Alcohol, Volatile Acidity, Citric Acid, Sulphates, pH, etc.) and Wine Type selector (Red / White).
- **Interactive Feature Importance:** Visual ranking displaying which chemical attributes influence the prediction the most (top drivers: Alcohol, Volatile Acidity, Density).
- **Fast RESTful API:** Endpoint for programmatic scoring (`POST /api/predict`).
- **Serverless & Cloud Ready:** Configured for Vercel Serverless Functions (`@vercel/python`) as well as Docker / Render / Heroku via `Procfile`.

---

## 🚀 Live Demo

Visit [wine-quality-app-eight.vercel.app](https://wine-quality-app-eight.vercel.app) to test wine samples live.

### Sample Test Cases:

| Wine Type | Alcohol | Volatile Acidity | Citric Acid | Sulphates | Predicted Quality | Probability Good |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Premium Red** | 12.8% | 0.25 g/L | 0.40 g/L | 0.80 g/L | **GOOD** | ~90.0% |
| **Low Grade Red** | 9.4% | 0.70 g/L | 0.00 g/L | 0.45 g/L | **POOR** | ~4.0% |

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, Flask 3.0, Flask-CORS
- **Machine Learning:** Scikit-learn (Random Forest Classifier), Pandas, NumPy, Joblib
- **Frontend:** Vanilla HTML5, Modern High-Contrast CSS, Responsive Layout, Fetch API
- **Deployment:** Vercel (`vercel.json`), Render (`render.yaml`), Gunicorn (`Procfile`)

---

## 💻 Local Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ITZRAHULPLAYZ/wine-quality-test.git
   cd wine-quality-test/Wine-Quality-Web-App
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:5000`.

---

## 📡 API Reference

### 1. Predict Wine Quality
- **Endpoint:** `POST /api/predict`
- **Request Body (JSON):**
  ```json
  {
    "fixed acidity": 8.0,
    "volatile acidity": 0.25,
    "citric acid": 0.4,
    "residual sugar": 2.0,
    "chlorides": 0.04,
    "free sulfur dioxide": 30.0,
    "total sulfur dioxide": 90.0,
    "density": 0.993,
    "pH": 3.3,
    "sulphates": 0.8,
    "alcohol": 12.8,
    "is_white": 0
  }
  ```
- **Response (JSON):**
  ```json
  {
    "prediction": "GOOD",
    "probability_good": 90.0,
    "confidence": 90.0,
    "is_good": true
  }
  ```

### 2. Feature Importance
- **Endpoint:** `GET /api/feature-importance`
- **Response (JSON):** Returns sorted Gini importance values for all 12 model features.

### 3. Health Check
- **Endpoint:** `GET /api/health`
- **Response (JSON):** `{"status": "ok", "model_loaded": true}`
