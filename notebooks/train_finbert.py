from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

# Load your CSV training data
df = pd.read_csv("data/finbert_train.csv")  # Make sure this path matches
dataset = Dataset.from_pandas(df)

# Tokenizer & model (FinBERT pretrained on financial text)
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.train_test_split(test_size=0.2)

# Load model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Training args
training_args = TrainingArguments(
    output_dir="./models/finbert-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer
)

# Train!
trainer.train()

# Save model
trainer.save_model("models/finbert-finetuned")
print("FinBERT fine-tuning complete and saved.")
