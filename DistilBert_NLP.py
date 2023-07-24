import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch.optim as optim
from datasets import load_dataset
import pandas as pd
from torch.nn import CrossEntropyLoss
import time


# Load the emotion dataset from Hugging Face
dataset = load_dataset("dair-ai/emotion", split="train")

# Extract the texts and labels from the dataset
texts = dataset["text"]
labels = dataset["label"]

# Map the integer labels to their corresponding string labels
label_string_map = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
labels = [label_string_map[label] for label in labels]

# Convert the labels from strings to integers
label_map = {"sadness": 0, "joy": 1, "love": 2, "anger": 3, "fear": 4, "surprise": 5}
labels = [label_map[label] for label in labels]

# Preprocess the texts
preprocessed_texts = []
for text in texts:
    preprocessed_text = text.lower()
    preprocessed_texts.append(preprocessed_text)

# Split the data into train, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(preprocessed_texts, labels, test_size=0.1, stratify=labels, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1111, stratify=train_labels, random_state=42)

# Convert the data to Pandas dataframes
train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
val_df = pd.DataFrame({"text": val_texts, "label": val_labels})
test_df = pd.DataFrame({"text": test_texts, "label": test_labels})

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Define the SentimentDataset class
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {"input_ids": encoding["input_ids"].squeeze(), "attention_mask": encoding["attention_mask"].squeeze(), "label": label}

# Create DataLoaders
train_dataset = SentimentDataset(train_df, tokenizer, max_length=256)
val_dataset = SentimentDataset(val_df, tokenizer, max_length=256)
test_dataset = SentimentDataset(test_df, tokenizer, max_length=256)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_map))

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to training mode
model.train()

# Initialize the optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()

# Number of training epochs
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward propagation
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

        # Measure the time taken for a single batch
        if batch_idx == 0:
            start_time = time.time()
        elif batch_idx == 1:
            time_per_batch = time.time() - start_time
            print(f"Time per batch: {time_per_batch:.4f} seconds")
            print(f"Estimated time per epoch: {time_per_batch * len(train_loader):.4f} seconds")

    # Compute the average training loss for this epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Training loss: {epoch_loss:.4f}")

    # Evaluate the model on the validation set
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    print(f"Validation accuracy: {accuracy:.4f}")

# Save the model
model.save_pretrained("my_sentiment_bot")
