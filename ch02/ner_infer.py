# PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 python ner_infer.py

from pathlib import Path
import warnings

import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

# Disable future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


# Set device to GPU if available, otherwise fall back to CPU
# Prioritize Apple Silicon MPS, then CUDA, then CPU
device = torch.device("cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon) for inference.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (GPU) for inference.")
else:
    print("Using CPU for inference.")


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


modelPath = Path("ner_model_complete.pth")

# Load the model
print(f"Loading checkpoint from {modelPath}...", flush=True)
config = BertConfig.from_pretrained(
    "bert-base-uncased",
    num_labels=len(ner_tag_map),
    id2label=ner_tag_map_rev,
    label2id=ner_tag_map,
)
model = BertForTokenClassification(config)
model.load_state_dict(torch.load(modelPath, map_location=device))
model.to(device)
model.eval()
print("Model loaded and ready for inference.", flush=True)


# Function to detect named entities in a sentence
def detect_named_entities(sentence, model, tokenizer, ner_tag_map_rev):
    # Tokenize the input sentence into words
    tokens = sentence.split()

    # Tokenize the input tokens using the tokenizer
    tokenized_inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    # Send to device (CPU or GPU)
    tokenized_inputs = {key: val.to(device) for key, val in tokenized_inputs.items()}

    # Predict using the model
    model.eval()
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Convert predictions back to tag names
    predicted_tags = [ner_tag_map_rev[idx] for idx in predictions[0].cpu().tolist()]

    # Extract original tokens and corresponding predicted labels
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])

    # Align tokens and labels, skipping special tokens
    result = []
    for token, tag in zip(tokens, predicted_tags):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        elif token.startswith("##") and result:
            result[-1][0] += token[2:]
        else:
            result.append([token, tag])

    # Return one label per token and omit non-entity tokens
    return {token: tag for token, tag in result if tag != "O"}


# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# List of test sentences
test_sentences = [
    "Dr. Alice Smith from Stanford University will attend the conference on July 20th.",
    "Google announced a new product in San Francisco last Friday.",
    "John Doe donated $5000 to the Red Cross during the 2020 pandemic.",
    "The Eiffel Tower is one of the most famous landmarks in Paris, France.",
]


# Function to test multiple sentences
def test_ner_model(sentences, model, tokenizer, ner_tag_map_rev):
    for idx, sentence in enumerate(sentences):
        print(f"\nSentence {idx + 1}: {sentence}")
        detected_entities = detect_named_entities(
            sentence, model, tokenizer, ner_tag_map_rev
        )
        print("Detected Named Entities:")
        for entity, entity_type in detected_entities.items():
            print(f"  {entity}: {entity_type}")


# Run the test
test_ner_model(test_sentences, model, tokenizer, ner_tag_map_rev)
