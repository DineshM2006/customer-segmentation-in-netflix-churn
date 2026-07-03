# 🏦 Customer Segmentation in netflix Churn using Hybrid Clustering

## 📌 Overview

This project focuses on **Customer Segmentation** for a bank using **Hybrid Clustering** techniques. The application groups customers with similar characteristics, enabling banks to better understand customer behavior, improve marketing strategies, and enhance customer retention.

The project is developed using **Python**, **Scikit-learn**, and **Streamlit**, and is deployed online for interactive use.

---

## 🚀 Live Demo

🌐 **Streamlit Application**

https://customer-segmentation-in-bank-churn-lvpllacq3stovdmjcffzmd.streamlit.app/

---

## 🎯 Project Objectives

* Analyze customer netfix data.
* Perform data preprocessing and feature engineering.
* Apply Hybrid Clustering to identify customer segments.
* Visualize customer groups.
* Deploy the model as an interactive Streamlit web application.

---

## 📂 Dataset

The project uses a **Bank Customer Churn Dataset** containing customer demographic and banking information.

Typical features include:

* Age
* Gender
* Credit Score
* Balance
* Estimated Salary
* Tenure
* Number of Products
* Active Member Status
* Geography
* Other banking-related attributes

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Missing Value Handling
4. Label Encoding
5. Feature Scaling
6. Hybrid Clustering
7. Cluster Assignment
8. Model Serialization
9. Streamlit Deployment

---

## 🤖 Hybrid Clustering Approach

The project combines multiple clustering techniques to improve segmentation performance:

* BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
* MiniBatch K-Means

This approach is computationally efficient for large datasets while maintaining high-quality customer segmentation.

---

## 📊 Features

* Upload customer dataset (CSV)
* Automatic preprocessing
* Customer segmentation using Hybrid Clustering
* View clustered dataset
* Download prediction results
* Interactive Streamlit interface

---

## 📁 Project Structure

```
Customer-Segmentation/

│── app.py
│── requirements.txt
│── hybrid_model.pkl
│── scaler.pkl
│── clustered_dataset.csv
│── README.md
│── notebook.ipynb
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Customer-Segmentation.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📈 Output

The application assigns every customer to a cluster such as:

| Customer   | Cluster |
| ---------- | ------- |
| Customer 1 | 0       |
| Customer 2 | 1       |
| Customer 3 | 2       |
| Customer 4 | 3       |

These clusters help identify customer groups with similar banking behaviors.

---

## 🌟 Future Enhancements

* Cluster visualization dashboard
* SHAP-based explainability
* Customer retention recommendations
* Interactive analytics dashboard
* Automatic optimal cluster selection

---

## 👨‍💻 Author

**Dinesh M**

---

## 📜 License

This project is intended for educational and academic purposes.
