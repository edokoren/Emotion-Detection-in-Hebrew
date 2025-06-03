# Emotion Detection in Hebrew Texts using HeBERT

This project is part of the final assignment for the Natural Language Processing (NLP) course at [Your University Name].  
It focuses on emotion classification in Hebrew, leveraging the `avichr/heBERT` model with fine-tuning for our custom dataset.

## 🎯 Project Objective

To train a Hebrew emotion classifier that can categorize sentences into one of six emotions:  
**joy, sadness, anger, fear, love, surprise**

The model is fine-tuned on a translated version of the [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/praveengovi/emotions), originally in English, and preprocessed into Hebrew for domain alignment.

## 🛠 Tools and Technologies

- Python
- PyTorch
- HuggingFace Transformers
- Google Colab
- scikit-learn
- Matplotlib / Seaborn (for visualization)

## 📁 Project Structure


├── train_1000.txt                # Training data (translated to Hebrew)
├── heb_emotion_classifier.ipynb # Colab notebook with full pipeline
├── model/                       # Saved model and tokenizer
├── inference_examples/          # Inference and log images
└── README.md                    # Project description

## 📊 Methodology

### 1. Data Preparation
- Translated original dataset to Hebrew using Google Translate API.
- Encoded labels with `LabelEncoder`.
- Tokenized the text with `HeBERT tokenizer`.

### 2. Model Training
We fine-tuned the `avichr/heBERT` model on CPU using PyTorch (manual training loop).  
Training was performed over **3 epochs** with a batch size of 16.

**Training Log Snapshot**  
![Training Log](inference_examples/training_log.png)

### 3. Inference
After training, we evaluated the model on example inputs and visualized prediction accuracy.

**Example Inference Output**  
![Inference](inference_examples/inference_output.png)

### 4. Evaluation Metrics
We computed the following metrics on the test set:

- Accuracy
- Precision
- Recall
- F1-Score

**Emotion-wise Evaluation**  
![Emotion Metrics](inference_examples/emotion_metrics_graph.png)

## 🚀 Running the Project

1. Clone this repo:

git clone https://github.com/your-username/HeBERT-Emotion-Classification.git
cd HeBERT-Emotion-Classification


2. Install dependencies:

pip install -r requirements.txt


3. Run the notebook:
Use Google Colab or Jupyter:

jupyter notebook heb_emotion_classifier.ipynb


## 👥 Authors

- Yotam Hasid  
- Amit Keinan  
- Edo Koren

## 📌 Acknowledgements

- [avichr/heBERT](https://huggingface.co/avichr/heBERT)
- HuggingFace Transformers
- Original Kaggle Dataset by Praveen G

---

This project was submitted as part of the final NLP course project in June 2025.
