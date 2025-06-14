# Emotion Detection in Hebrew using HeBERT

This project implements a fine-tuned version of [HeBERT](https://huggingface.co/avichr/heBERT), a BERT-based language model for Hebrew, to classify sentences into six emotional categories:

> `joy`, `sadness`, `anger`, `fear`, `love`, and `surprise`

---

## 📦 Project Structure

![image](https://github.com/user-attachments/assets/9960f3e3-fd9a-4ffa-aa33-4195790ba6fa)


---

## 🛠️ How to Run

> Requirements: Python 3.8+, PyTorch, HuggingFace Transformers, scikit-learn, matplotlib, seaborn

### 🔹 1. Train the Model

python train_model.py


This will fine-tune HeBERT on the emotion dataset and save the model in `hebert_emotion_trained/`.

### 🔹 2. Run Inference

python inference.py


This will:
- Predict emotions on `test.txt`
- Compute accuracy and F1 scores
- Save:
  - `confusion_matrix.png`
  - `f1_scores.png`

---

## 📊 Results (sample of 50 test sentences)
- ✅ Accuracy: 0.3400
- 📉 **F1 Score per Emotion:**

  ![F1 Scores](f1_scores.png)

- 🔄 **Confusion Matrix:**

  ![Confusion Matrix](confusion_matrix.png)

---

## 📁 Dataset Format

### `train_balanced_750.txt` and `test.txt` structure:
Each line contains a sentence and its emotion label separated by a semicolon `;`

**Example:**

I feel so connected to you;love
Why did this happen to me?;anger
I'm smiling from ear to ear;joy


---

## ✍️ Author

- **Yotam Hassid** 
- **Amit Keinan**
- **Edo Koren**
  NLP Course Project – 2025  
  HIT – Holon Institute of Technology

---

## 📌 Tags

`#NLP` `#HeBERT` `#EmotionDetection` `#Hebrew` `#Transformers` `#FineTuning`
