import argparse
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune the encoder of BART")
    parser.add_argument(
        "--train_file", type=str, default=None)
    parser.add_argument(
        "--validation_file", type=str, default=None)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    raw_datasets = load_dataset('csv', data_files={'train': args.train_file, 'validation': args.validation_file})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["note"],
            max_length=1024,
            truncation=True,
        )
        labels = tokenizer(
            examples["note"], max_length=1024, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(['notes'])

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    # freeze the decoder of the model, if not already frozen
    # for param in model.model.decoder.parameters():
    #     param.requires_grad = False

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
    )
    trainer.train()
    trainer.save_model(args.output_dir) 