import json
import os
import subprocess

# Step 1: Define training arguments with detailed descriptions
args = dict(
    stage="sft",  # Stage of training: 'sft' means supervised fine-tuning (instruction tuning)
    do_train=True,  # Flag to enable training
    model_name_or_path="unsloth/llama-3-8b-Instruct-bnb-4bit",
    # Path or model name from Hugging Face to use as the base model
    dataset="identity",  # Name of your dataset or path to your dataset (must be in supported format like JSON or JSONL)
    template="llama3",  # Prompt template style to format instruction/input/output (e.g., llama3, alpaca, chatml)
    finetuning_type="lora",  # Type of parameter-efficient fine-tuning method (e.g., lora, full, qlora)
    lora_target="q_proj,v_proj",  # Target layers to apply LoRA on (commonly attention projection layers)

    output_dir="llama3_lora_identity",  # Directory to save the trained model and checkpoints
    per_device_train_batch_size=2,  # Batch size per GPU
    gradient_accumulation_steps=2,  # Accumulate gradients over this many steps before backpropagation
    lr_scheduler_type="cosine",  # Learning rate scheduler type (e.g., linear, cosine, constant)
    warmup_ratio=0.1,  # Fraction of total steps used for learning rate warm-up

    logging_steps=1,  # Log training metrics every N steps
    save_steps=10,  # Save model checkpoint every N steps
    save_total_limit=2,  # Maximum number of saved checkpoints (older ones deleted)

    learning_rate=2e-5,  # Initial learning rate
    num_train_epochs=20,  # Number of full training epochs
    max_samples=50,  # Max number of samples to use from the dataset (useful for debugging or quick runs)
    max_grad_norm=1.0,  # Gradient clipping threshold to prevent exploding gradients

    loraplus_lr_ratio=4.0,  # Optional ratio to scale LoRA layers' learning rate compared to base learning rate
    fp16=True,  # Use 16-bit floating point precision for training (saves memory, speeds up training)
    report_to="none",  # Disable logging to external tools (like WandB or TensorBoard)
)

# Step 2: Save training config
with open("llama3_lora_identity.json", "w", encoding="utf-8") as f:
    json.dump(args, f, indent=2)

# Step 3: Change to LLaMA-Factory directory (update this path to match your setup)
os.chdir("/LLaMA-Factory")

# Step 4: Run training using the LLaMA-Factory CLI
subprocess.run(["llamafactory-cli", "train", "llama3_lora_identity.json"])


