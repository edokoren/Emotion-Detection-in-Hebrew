
# שלב 1: טעינת ספריות
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# שלב 2: טעינת המודל המאומן
model = AutoModelForSequenceClassification.from_pretrained("./hebert_emotion_trained")
tokenizer = AutoTokenizer.from_pretrained("./hebert_emotion_trained")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# שלב 3: טעינת קובץ test_real_hard_150.txt
test_df = pd.read_csv("test_real_hard_150.txt", sep="\t", header=None, names=["text", "label"])
label2id = model.config.label2id
id2label = model.config.id2label
test_df = test_df[test_df["label"].isin(label2id.keys())]
test_df["label_id"] = test_df["label"].map(label2id)

# שלב 4: חיזוי
texts = test_df["text"].tolist()
true_labels = test_df["label_id"].tolist()
predicted_labels = []

batch_size = 16
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encodings = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
        predicted_labels.extend(preds)

# שלב 5: מדדים
acc = accuracy_score(true_labels, predicted_labels)
print(f"✅ דיוק כולל (Accuracy): {acc:.4f}")

# F1 לכל רגש
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=list(id2label.keys()))
for i, label in id2label.items():
    print(f"{label}: Precision={precision[i]:.2f}, Recall={recall[i]:.2f}, F1={f1[i]:.2f}")

# מטריצת בלבול
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=[id2label[i] for i in range(len(id2label))],
            yticklabels=[id2label[i] for i in range(len(id2label))])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("🔷 Confusion Matrix")
plt.savefig("confusion_matrix2.png", dpi=150)
plt.close()

# גרף F1
plt.figure(figsize=(8, 5))
plt.bar([id2label[i] for i in range(len(id2label))], f1, color='orange')
plt.title("🎯 F1 Score by Emotion")
plt.ylim(0, 1)
plt.ylabel("F1 Score")
plt.grid(axis='y')
plt.savefig("f1_scores2.png", dpi=150)
plt.close()

print("✅ גרפים נשמרו: confusion_matrix2.png, f1_scores2.png")
