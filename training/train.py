"""Fine-tune T5-small on the synthetic robot-control dataset.

Run from anywhere: ``python training/train.py`` (paths are relative to this file).
"""

import json
import os
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "t5-small"
OUTPUT_DIR = os.path.join(BASE_DIR, "trained_model")
DATA_FILE = os.path.join(BASE_DIR, "robot_training_data.jsonl")
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3


def load_data(path):
    inputs, outputs = [], []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            inputs.append(item["input"])
            outputs.append(item["output"])
    return inputs, outputs


def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=examples["output"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    print("Loading data...")
    inputs, outputs = load_data(DATA_FILE)
    print(f"Loaded {len(inputs)} examples")

    dataset = Dataset.from_dict({"input": inputs, "output": outputs})

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["<tree>", "<rock>", "<house>", "<person>", "<car>", "<bush>", "<lamp>", "<sign>"])

    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "output"],
    )

    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        report_to="none",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Training...")
    trainer.train()

    print(f"Saving to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done!")


if __name__ == "__main__":
    main()
