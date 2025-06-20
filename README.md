# Emotion Detection in Hebrew 🇮🇱

Final NLP Project — Emotion Classification in Hebrew Texts  
**By Yotam Hasid, Amit Keinan, Edo Koren | HIT | 2025** 

---

## 🔖  Description

Emotion Detection in Hebrew - NLP Final Project using HeBERT, achieving 70% accuracy on realistic test set. 

---

## 🎯 Project Description

This project implements **Emotion Detection in Hebrew**  
using Fine-Tuning of the model `avichr/heBERT`.

The project includes:

✅ Fine-Tuning on custom dataset  
✅ Evaluation on a realistic hard test set  
✅ Confusion Matrix + F1 Score Plot  
✅ `train_model.py` script  
✅ `inference.py` script

---

## 🧠 Model

- **Base model**: [avichr/HeBERT](https://huggingface.co/avichr/heBERT)
- **Fine-tuned epochs**: 4
- **Train set**: 1000 high-quality labeled examples
- **Test set**: 150 realistic challenging examples

---

## 📊 Results on test_real_hard_150.txt

**Overall Accuracy**: `0.70` ✅

| Emotion    | Precision | Recall | F1  |
|------------|-----------|--------|-----|
| anger      | 0.71      | 0.83   | 0.77 |
| fear       | 0.75      | 0.43   | 0.55 |
| joy        | 0.62      | 0.71   | 0.67 |
| love       | 1.00      | 0.50   | 0.67 |
| sadness    | 0.58      | 1.00   | 0.74 |
| surprise   | 0.83      | 0.71   | 0.77 |

---

## 🔍 Confusion Matrix

![Confusion Matrix](confusion_matrix2.png)

---

## 🎭 F1 Score by Emotion

![F1 Score](f1_scores2.png)

---

## 🗂 Project Structure

| File / Folder               | Description                          |
|-----------------------------|--------------------------------------|
| train_balanced_1000.txt      | Training dataset (1000 sentences)     |
| test_real_hard_150.txt       | Real hard test set (150 sentences)    |
| hebert_emotion_trained/      | Trained model directory               |
| train_model.py               | Training script                       |
| inference.py                 | Inference & evaluation script         |
| requirements.txt             | Python dependencies                   |
| README.md                    | Project description                   |

---

## 🏷️ Target Emotions

The model classifies each sentence into one of these 6 emotions:

1. joy  
2. sadness  
3. anger  
4. fear  
5. love  
6. surprise

---

## 🚀 Installation


pip install -r requirements.txt


---

## 🎓 Running Training


python train_model.py


---

## 🔍 Running Inference + Results


python inference.py


---

## 🎓 Conclusion

- Model achieves **~70% accuracy** on challenging test — great for real-life noisy text.
- Further improvements can include:
    - Expanding the train set
    - Adding more edge cases and mixed emotions
    - Applying data augmentation

---

## 👨🏻‍💻 Authors

* Yotam Hasid  
* Amit Keinan  
* Edo Koren  

---

## 📜 License

MIT License — see LICENSE file for details.

---
