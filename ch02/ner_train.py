# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python ner_train.py

import os
from pathlib import Path
import warnings

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

NUM_EPOCHS = 3
BATCH_SIZE = 8
PROGRESS_STEPS = 10


# Set device to GPU if available, otherwise fall back to CPU
# Prioritize Apple Silicon MPS, then CUDA, then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for training.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for training.")
else:
    print("Using CPU for training.")


# NER tag mapping: Maps NER tags to unique integer indices
ner_tag_map = {
    "B-AFFILIATION": 0,
    "I-AFFILIATION": 1,
    "B-ANATOMICAL": 2,
    "I-ANATOMICAL": 3,
    "B-ATTRIBUTE": 4,
    "I-ATTRIBUTE": 5,
    "B-BRANDS": 6,
    "I-BRANDS": 7,
    "B-DATE": 8,
    "I-DATE": 9,
    "B-DOCUMENT": 10,
    "I-DOCUMENT": 11,
    "B-DRUG": 12,
    "I-DRUG": 13,
    "B-DURATION": 14,
    "I-DURATION": 15,
    "B-EVENT": 16,
    "I-EVENT": 17,
    "B-FAMILY_NAME": 18,
    "I-FAMILY_NAME": 19,
    "B-GIVEN_NAME": 20,
    "I-GIVEN_NAME": 21,
    "B-LOCATION": 22,
    "I-LOCATION": 23,
    "B-MEDICAL-CONDITION": 24,
    "I-MEDICAL-CONDITION": 25,
    "B-MONEY": 26,
    "I-MONEY": 27,
    "B-NAME": 28,
    "I-NAME": 29,
    "B-NUMERIC": 30,
    "I-NUMERIC": 31,
    "B-ORGANIZATION": 32,
    "I-ORGANIZATION": 33,
    "B-OTHER": 34,
    "I-OTHER": 35,
    "B-PRICE": 36,
    "I-PRICE": 37,
    "B-STATUS": 38,
    "I-STATUS": 39,
    "B-TIME": 40,
    "I-TIME": 41,
    "B-MISC": 42,
    "I-MISC": 43,
    "O": 44,
}

# Reverse mapping: Maps indices back to NER tags
ner_tag_map_rev = {v: k for k, v in ner_tag_map.items()}


# Load all data files from the specified directory recursively
def load_all_data_from_directory(directory):
    all_sentences = []
    all_tags = []

    # Traverse directory to find all files
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Loading data from {file_path}")
            sentences, tags = load_data(file_path)
            all_sentences.extend(sentences)
            all_tags.extend(tags)

    return all_sentences, all_tags


# Load individual data file in CoNLL format
def load_data(file_path):
    sentences = []
    tags = []
    sentence = []
    tag = []

    with open(file_path, "r") as file:
        for line in file:
            if line.strip() == "":
                if sentence:  # Ensure empty sentences aren't added
                    sentences.append(sentence)
                    tags.append(tag)
                sentence = []
                tag = []
            else:
                token, ner_tag = line.strip().split("\t")
                sentence.append(token)
                tag.append(ner_tag.split("|")[0])  # Keep one label per token

    # Add the last sentence if file doesn't end with a newline
    if sentence:
        sentences.append(sentence)
        tags.append(tag)

    return sentences, tags


# Tokenization function, aligning tokens with labels
def tokenize_and_preserve_labels(sentences, tags, tokenizer, ner_tag_map):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Ignored by the loss function
            else:
                label_ids.append(ner_tag_map[label[word_idx]])
        labels.append(label_ids)

    return tokenized_inputs, labels


# Custom Dataset class for lazy loading
class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, ner_tag_map):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.ner_tag_map = ner_tag_map

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag = self.tags[idx]

        # Tokenize sentence and preserve labels
        tokenized_inputs, label = tokenize_and_preserve_labels(
            [sentence], [tag], self.tokenizer, self.ner_tag_map
        )

        return {
            "input_ids": tokenized_inputs["input_ids"][0],
            "attention_mask": tokenized_inputs["attention_mask"][0],
            "labels": torch.tensor(label[0], dtype=torch.long),
        }


# Collate function to pad sequences within each batch:
# - Padding: Ensures all sequences in a batch are the same length.
# - Masking: Uses -100 for labels of padded tokens to ignore them during loss computation.
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
        "attention_mask": pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        ),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }


# Create DataLoader for training and validation
def create_dataloader(dataset, batch_size=16):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )


modelPath = Path("ner_model_complete.pth")
if not modelPath.exists():
    print(f"No checkpoint found at {modelPath}. Starting training.", flush=True)

    # Directory containing the data files
    data_directory = "CoNLL-2003"

    # Load data from the directory
    print(f"Loading training data from {data_directory}/ ...", flush=True)
    all_sentences, all_tags = load_all_data_from_directory(data_directory)
    print(f"Loaded {len(all_sentences):,} sentences.", flush=True)

    # Split data into train and validation sets (80% train, 20% validation)
    train_sentences, val_sentences, train_tags, val_tags = train_test_split(
        all_sentences, all_tags, test_size=0.2, shuffle=True, random_state=42
    )
    print(
        f"Training examples: {len(train_sentences):,} | "
        f"Validation examples: {len(val_sentences):,}",
        flush=True,
    )

    # Initialize BERT tokenizer
    print("Loading bert-base-uncased tokenizer...", flush=True)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Create Dataset objects for training and validation
    train_dataset = NERDataset(train_sentences, train_tags, tokenizer, ner_tag_map)
    val_dataset = NERDataset(val_sentences, val_tags, tokenizer, ner_tag_map)

    # Create DataLoader objects
    train_dataloader = create_dataloader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = create_dataloader(val_dataset, batch_size=BATCH_SIZE)
    print(
        f"Training batches: {len(train_dataloader):,} | "
        f"Validation batches: {len(val_dataloader):,} | "
        f"Batch size: {BATCH_SIZE}",
        flush=True,
    )

    # Define the BERT model for single-label token classification
    print("Loading bert-base-uncased model for token classification...", flush=True)
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=len(ner_tag_map),
        id2label=ner_tag_map_rev,
        label2id=ner_tag_map,
    )
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased", config=config
    )
    model.to(device)  # Move model to device
    print(f"Model moved to {device}.", flush=True)

    # Define optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_steps = len(train_dataloader) * NUM_EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs.", flush=True)
    progress_interval = max(1, len(train_dataloader) // PROGRESS_STEPS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} started.", flush=True)
        for batch_idx, batch in enumerate(train_dataloader, start=1):
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_masks = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks,
                labels=batch_labels,
            )
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            scheduler.step()

            if batch_idx == 1 or batch_idx % progress_interval == 0 or batch_idx == len(train_dataloader):
                avg_loss_so_far = total_loss / batch_idx
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"Batch {batch_idx:,}/{len(train_dataloader):,} | "
                    f"Avg loss: {avg_loss_so_far:.4f}",
                    flush=True,
                )

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} finished. Avg loss: {avg_loss:.4f}", flush=True)

    # Evaluate the model
    print("\nStarting validation.", flush=True)
    model.eval()
    all_preds = []
    all_labels = []
    val_progress_interval = max(1, len(val_dataloader) // PROGRESS_STEPS)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader, start=1):
            batch_input_ids = batch["input_ids"].to(device)
            batch_attention_masks = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(
                input_ids=batch_input_ids, attention_mask=batch_attention_masks
            )
            predictions = torch.argmax(outputs.logits, dim=-1)

            active_tokens = batch_labels.view(-1) != -100
            active_preds = predictions.view(-1)[active_tokens]
            active_labels = batch_labels.view(-1)[active_tokens]

            all_preds.extend(active_preds.cpu().tolist())
            all_labels.extend(active_labels.cpu().tolist())

            if batch_idx == 1 or batch_idx % val_progress_interval == 0 or batch_idx == len(val_dataloader):
                print(
                    f"Validation batch {batch_idx:,}/{len(val_dataloader):,}",
                    flush=True,
                )

    # Calculate metrics
    f1 = f1_score(all_labels, all_preds, average="micro")
    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")

    print(f"Validation Precision: {precision:.4f}", flush=True)
    print(f"Validation Recall:    {recall:.4f}", flush=True)
    print(f"Validation F1 Score:  {f1:.4f}", flush=True)

    # After training is complete, save the model's state_dict
    print(f"Saving checkpoint to {modelPath}...", flush=True)
    torch.save(model.state_dict(), modelPath)

    # You can also save the entire model if needed, though saving the state_dict is preferred
    # torch.save(model, "ner_model_single_label_complete.pth")
    print("Training complete.", flush=True)
else:
    print(f"Checkpoint found at {modelPath}. Skipping training.", flush=True)
