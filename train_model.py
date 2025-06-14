import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd

label2id = {'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3, 'love': 4, 'surprise': 5}
id2label = {v: k for k, v in label2id.items()}

# Load data
df = pd.read_csv("train_balanced_750.txt", sep=";", header=None, names=["text", "label"])
df = df[df["label"].isin(label2id.keys())]
df["label_id"] = df["label"].map(label2id)

# Dataset class
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
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Tokenizer & DataLoader
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
dataset = EmotionDataset(df["text"].tolist(), df["label_id"].tolist(), tokenizer)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "avichr/heBERT", num_labels=6, id2label=id2label, label2id=label2id
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_dataloader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())
    print(f"Average Loss: {total_loss / len(train_dataloader):.4f}")

# Save model
model.save_pretrained("hebert_emotion_trained")
tokenizer.save_pretrained("hebert_emotion_trained")
print("✅ Model saved successfully.")
