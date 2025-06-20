
# שלב 1: טעינת ספריות
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# שלב 2: טעינת הדאטה החדש
df = pd.read_csv("train_balanced_1000.txt", sep=";", header=None, names=["text", "label"])
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["label"])
id2label = {int(i): label for i, label in enumerate(le.classes_)}
label2id = {v: k for k, v in id2label.items()}

# שלב 3: מחלקת Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# שלב 4: Tokenizer + DataLoader
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
dataset = EmotionDataset(df["text"].tolist(), df["label_id"].tolist(), tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# שלב 5: הגדרת המודל
model = AutoModelForSequenceClassification.from_pretrained(
    "avichr/heBERT",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# שלב 6: אימון
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 5

model.train()
for epoch in range(epochs):
    print(f"\n🔁 Epoch {epoch+1}")
    loop = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    total_loss = 0

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"📉 Loss ממוצע באפוק {epoch+1}: {avg_loss:.4f}")

# שלב 7: שמירת המודל המאומן
model.save_pretrained("hebert_emotion_trained")
tokenizer.save_pretrained("hebert_emotion_trained")
print("✅ המודל נשמר בהצלחה בתיקייה 'hebert_emotion_trained' 🎉")
