import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

label2id = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}
id2label = {v: k for k, v in label2id.items()}

# Load model
tokenizer = AutoTokenizer.from_pretrained("hebert_emotion_trained")
model = AutoModelForSequenceClassification.from_pretrained("hebert_emotion_trained")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load test set (limit to 50 for speed)
df = pd.read_csv("test.txt", sep=";", header=None, names=["text", "label"])
df = df[df["label"].isin(label2id.keys())].head(50)
df["label_id"] = df["label"].map(label2id)

texts = df["text"].tolist()
true_labels = df["label_id"].tolist()

# Predict in small batches
batch_size = 16
predicted_labels = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encodings = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        predicted_labels.extend(preds)

# Accuracy
acc = accuracy_score(true_labels, predicted_labels)
print(f"✅ Accuracy: {acc:.4f}")

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=label2id.keys(), yticklabels=label2id.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# F1 scores
_, _, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=list(label2id.values()), zero_division=0)
plt.figure(figsize=(6, 4))
plt.bar(label2id.keys(), f1)
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.title("F1 Score by Emotion")
plt.tight_layout()
plt.savefig("f1_scores.png")
plt.close()

print("📁 Saved: confusion_matrix.png, f1_scores.png")
