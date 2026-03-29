# Fake-Job-Detection

```markdown
# 🛡️ Fake Job Detection System

A web-based application that detects fraudulent job postings using an ensemble of advanced machine learning and transformer-based models. The system analyzes job descriptions, company details, and related metadata to classify jobs as **legitimate** or **fraudulent**.

---

## 📌 Features

- 🔍 Detects fake vs legitimate job postings  
- 🤖 Ensemble model (DeBERTa + Sentence Transformer + ML classifier)  
- 🌐 Web interface built with Django  
- 📊 Displays prediction confidence score  
- 🧠 NLP-based feature extraction  

---

## 🏗️ System Architecture

```

User Input (UI - Django)
↓
Data Preprocessing
↓
Feature Extraction
├── DeBERTa Model
├── Sentence Transformer (all-mpnet-base-v2)
↓
Ensemble Layer (ML Classifier)
↓
Prediction Output (Fake / Real + Confidence)
↓
Displayed on UI

```

---

## 🧪 Technologies Used

### 🖥️ Backend
- Python  
- Django  

### 🤖 Machine Learning / NLP
- Hugging Face Transformers  
- DeBERTa-v3-base  
- Sentence Transformers (all-mpnet-base-v2)  
- Scikit-learn  

### 📊 Data Handling
- Pandas  
- NumPy  

### 🌐 Frontend
- HTML, CSS (Django Templates)  

---

## 📂 Project Structure

```

Fake-Job-Detection/
│── app/
│   ├── models.py
│   ├── views.py
│   ├── forms.py
│   ├── ml/
│   │   ├── model.py
│   │   ├── preprocessing.py
│   │   ├── inference.py
│   ├── templates/
│   ├── static/
│
│── project/
│   ├── settings.py
│   ├── urls.py
│
│── manage.py
│── requirements.txt
│── README.md

````

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fake-job-detection.git
cd fake-job-detection
````

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Migrations

```bash
python manage.py migrate
```

### 5. Start the Server

```bash
python manage.py runserver
```

### 6. Open in Browser

```
http://127.0.0.1:8000/
```

---

## 📊 Model Details

| Component            | Description                       |
| -------------------- | --------------------------------- |
| DeBERTa-v3           | Contextual language understanding |
| Sentence Transformer | Semantic embeddings               |
| ML Classifier        | Final prediction (ensemble layer) |

---

## 📈 Output

* ✅ Legitimate Job
* ❌ Fraudulent Job
* 📊 Confidence Score (%)

---

## 📉 Dataset

* Labeled job postings:

  * Legitimate (0)
  * Fraudulent (1)

* Preprocessing steps:

  * Text cleaning
  * Tokenization
  * Embedding generation

---

## 🚀 Future Improvements

* 🔐 User authentication
* 📱 Mobile-friendly UI
* ☁️ Cloud deployment
* 📊 Dashboard for analytics
* 🔄 Real-time job scraping

---

## 🤝 Contribution

```bash
# Fork the repo
# Create a new branch
git checkout -b feature-name

# Commit changes
git commit -m "Added feature"

# Push
git push origin feature-name
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Your Name
GitHub: [https://github.com/your-username](https://github.com/your-username)
Email: [your-email@example.com](mailto:your-email@example.com)

```
```
