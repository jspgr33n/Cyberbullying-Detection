
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['tweet_text'], padding='max_length', truncation=True, max_length=128)

file_path = '../data/train_data.csv'  
dataset = load_dataset('csv', data_files={'train': file_path})

train_valid_split = dataset['train'].train_test_split(test_size=0.2)

train_dataset = train_valid_split['train']
valid_dataset = train_valid_split['test']

# test_dataset = test_valid['test']

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# tokenized_test = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
tokenized_valid.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

# tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])

training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=10_000,
    eval_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./BERT_trained_removed")
tokenizer.save_pretrained("./BERT_trained_removed")

# results = trainer.evaluate(tokenized_test)

