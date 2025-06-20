Emotion Detection in Hebrew 🇮🇱
This project implements an Emotion Detection model in Hebrew, using the HeBERT model from HuggingFace.
The model was fine-tuned on a custom dataset of 1000 examples (train_balanced_1000.txt), and tested on a challenging realistic test set of 150 examples (test_real_hard_150.txt).

🚀 Goals
Classify a sentence into one of 6 emotions:
anger, fear, joy, love, sadness, surprise.

Fine-tune a HeBERT model on a custom Hebrew dataset.

Evaluate on a realistic, noisy test set — not overfit.

🧠 Model
Base model: avichr/HeBERT
Fine-tuned epochs: 5
Train set: 1000 high-quality labeled examples
Test set: 150 realistic challenging examples
📊 Results on test_real_150.txt
Overall Accuracy: 0.70 ✅

Emotion	Precision	Recall	F1
anger	0.71	0.83	0.77
fear	0.75	0.43	0.55
joy	0.62	0.71	0.67
love	1.00	0.50	0.67
sadness	0.58	1.00	0.74
surprise	0.83	0.71	0.77

🔍 Confusion Matrix

![image](https://github.com/user-attachments/assets/8b1a187e-1479-48f3-b3cd-0d662d2bfb2f)


🎭 F1 Score by Emotion

![image](https://github.com/user-attachments/assets/684a7dfd-0200-4f5f-8ecb-ec9aed24dd4a)


💾 Files

![image](https://github.com/user-attachments/assets/22b5be7f-2ccc-493d-bc2d-9e7fdb57e544)


🎓 Conclusion
Model achieves ~70% accuracy on challenging test — great for real-life noisy text.
Further improvements can include:
Expanding the train set
Adding more edge cases and mixed emotions
Applying data augmentation
Project by:

Yotam Hasid
Amit Keinan
Edo Koren
