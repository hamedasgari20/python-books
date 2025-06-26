# 📁 FILE: pipeline_automation.py
# PURPOSE: This script fine-tunes a lightweight open-source LLM (google/flan-t5-small)
#          on a Stack Overflow Q&A dataset using Parameter-Efficient Fine-Tuning (PEFT)
#          with LoRA (Low-Rank Adaptation) to reduce memory usage.
# TOOL: Hugging Face Transformers, PEFT, Datasets
# USE CASE: Customizing foundation models for domain-specific tasks like Stack Overflow QA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict
import json

# 🧾 Step 1: Load JSON training and evaluation data
# --------------------------------------------------------------------------------------------------
# The dataset contains Python-related questions and answers from Stack Overflow,
# saved in train_data.json and eval_data.json after preprocessing.
# We load them as lists of dictionaries.

with open("train_data.json", "r") as f:
    train_data = json.load(f)

with open("eval_data.json", "r") as f:
    eval_data = json.load(f)

# 📦 Step 2: Convert JSON data into Hugging Face Dataset objects
# --------------------------------------------------------------------------------------------------
# Hugging Face's Dataset class allows efficient tokenization, mapping, and training.
# Each dataset (train/eval) is converted from a list of dicts to a Dataset object.

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# 🗂️ Step 3: Wrap datasets into a DatasetDict
# --------------------------------------------------------------------------------------------------
# DatasetDict provides a standard structure for holding multiple splits (e.g., train/test).
# It simplifies working with different parts of the dataset during training.

dataset = DatasetDict({
    "train": train_dataset,
    "test": eval_dataset
})

# 🤖 Step 4: Load tokenizer and base model
# --------------------------------------------------------------------------------------------------
# We use google/flan-t5-small as the foundation model — a small, fast, and powerful T5-based model.
# Tokenizer converts text into numerical tokens the model can understand.

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 🔧 Step 5: Configure LoRA for efficient fine-tuning
# --------------------------------------------------------------------------------------------------
# LoRA reduces the number of trainable parameters, saving memory and time.
# We apply it to attention layers ("q" for query, "v" for value).
# This is part of the PEFT (Parameter Efficient Fine-Tuning) strategy.

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

# Apply LoRA to the base model
model = get_peft_model(model, peft_config)

# 🧹 Step 6: Define tokenization function
# --------------------------------------------------------------------------------------------------
# This function processes input text and output text using the tokenizer.
# Inputs are truncated/padded to 512 tokens; outputs to 128 tokens.
# Labels are added to the model inputs under the key "labels".

def tokenize_function(examples):
    inputs = examples["input_text"]
    targets = examples["output_text"]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply tokenization to the entire dataset using map()
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# ⚙️ Step 7: Set up training arguments
# --------------------------------------------------------------------------------------------------
# These define how the model will be trained: where to save results, learning rate,
# batch size, logging, and integration with Hugging Face Hub.
training_args = TrainingArguments(
    output_dir="./results",              # Directory to save training outputs
    eval_strategy="epoch",               # Evaluate every epoch
    learning_rate=2e-4,                 # Learning rate for optimizer
    per_device_train_batch_size=2,       # Batch size per device
    num_train_epochs=3,                  # Number of full passes through the data
    weight_decay=0.01,                   # Regularization to prevent overfitting
    logging_dir="./logs",                # Logs for TensorBoard or other tools
    push_to_hub=True,                    # Push final model to Hugging Face Hub
    hub_model_id="stackoverflow-flan-t5-small",  # Model name on Hugging Face
    report_to="none"                     # Disable extra reporting tools for simplicity
)


# 🧑‍🏫 Step 8: Initialize the Trainer
# --------------------------------------------------------------------------------------------------
# The Trainer class handles the training loop, including optimization and evaluation.
# We pass in the model, training arguments, and datasets for training and evaluation.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

# 🚀 Step 9: Start training process
# --------------------------------------------------------------------------------------------------
# The model begins adapting to the Stack Overflow Python Q&A dataset.
# This improves its ability to answer programming questions accurately.
print("🚀 Starting training...")
trainer.train()

# 💾 Step 10: Save and push tokenizer to Hugging Face Hub
# --------------------------------------------------------------------------------------------------
# To allow others to use the model, we must also provide the tokenizer.
# We save it locally and push it to the same repository.tokenizer.save_pretrained(training_args.output_dir)
tokenizer.push_to_hub(training_args.hub_model_id)

# 🌐 Step 11: Push model to Hugging Face Hub
# --------------------------------------------------------------------------------------------------
# After training, the model is uploaded to Hugging Face Hub for sharing and deployment.
print("📦 Pushing model to Hugging Face Hub...")
trainer.push_to_hub()
print("✅ Model and tokenizer pushed successfully.")