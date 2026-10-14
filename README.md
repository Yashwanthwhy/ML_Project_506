# Wine Quality Prediction and Feature Importance Analysis Using Random Forest

<div align="center">

![KLH University](https://img.shields.io/badge/KL%20Deemed%20University-CSE-blue?style=for-the-badge&logo=university&logoColor=white)
![Course](https://img.shields.io/badge/Course-Machine%20Learning-green?style=for-the-badge)
![Team](https://img.shields.io/badge/Team-64-orange?style=for-the-badge)

</div>

---

🌐 **Live Web Application:** [https://wine-quality-app-eight.vercel.app](https://wine-quality-app-eight.vercel.app)

---

## 1. Abstract

> The determination of the quality of wine has been a challenging issue as it is performed through expert sensory panels, which are subjective and hard to automate. The proposed work attempts to determine wine quality through a supervised machine learning approach by applying a Random Forest algorithm and determining the importance of physicochemical factors (alcohol, volatile acidity, citric acid, sulphates, pH) for predicting the wine quality from the UCI Wine Quality dataset.
>
> Apart from the predictive performance of the model, the importance of each factor will be analyzed based on the Gini importance measure, and the results will be validated against scientific knowledge about wine production. The performance of the classification model will be measured based on accuracy, precision, recall, F1-score, and a confusion matrix using k-fold cross-validation.
>
> Thus, by combining the predictive performance of the model with its explainability and feature level importance measures, the research illustrates how ensemble learning approaches could become more than just black box models in the context of practical tasks.

---

## 2. Team Details

| Roll Number | Name of the Student |
| :--- | :--- |
| **2520030305** | Sura.Rahul Reddy |
| **2520030506** | Yashwanth |

---

## 3. Frameworks / Platforms / Software Used

* **Language:** Python 3
* **Data Handling & Preprocessing:** Pandas, NumPy
* **Machine Learning & Evaluation:** Scikit-learn (Random Forest Classifier, train-test split, Stratified cross-validation, evaluation metrics)
* **Data Visualization:** Matplotlib, Seaborn (feature importance plotting, confusion matrix, ROC curves)
* **Web Framework & API:** Flask, Flask-CORS
* **Web Deployment:** Vercel Serverless Functions
* **Development Environment:** Jupyter Notebook, Google Colab, VS Code
* **Dataset Source:** Wine Quality Dataset (UCI Machine Learning Repository / Kaggle)

---

## 4. Live Web Application

An interactive web UI has been built and deployed live for real-time wine quality testing:
- **URL:** [https://wine-quality-app-eight.vercel.app](https://wine-quality-app-eight.vercel.app)
- **Features:** 11 chemical sliders, wine type selector, instant classification (**GOOD** / **POOR**), probability breakdown, and dynamic feature importance rankings.

---

## 📁 Repository Structure

```text
├── Project - Wine Quality Prediction and Feature Importance Analysis Using Random Forest/
│   ├── DataSet/
│   │   └── i will upload
│   ├── i will upload
│   ├── will upload
│   ├── Abstract.md
│   └── Wine quality random forest.ipynb
├── Wine-Quality-Web-App/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   ├── app.py
│   ├── train.py
│   ├── requirements.txt
│   ├── vercel.json
│   ├── Procfile
│   ├── render.yaml
│   ├── wine_rf_model.joblib
│   └── README.md
├── .gitignore
└── README.md
```
