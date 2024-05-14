import json
from transformers import T5ForConditionalGeneration, T5Tokenizer

train_data_path = "C:/Users/akash/Programming/eg_hackthon/documentaion/data.json"
tokenizer = T5Tokenizer.from_pretrained("t5-small")
with open(train_data_path, "r") as f:
    data = json.load(f)


def preprocess_documentation(doc_dict):
    doc_string = "\n".join(
        [f"{key.title().replace('_', ' ')}: {value}" for key, value in doc_dict.items()]
    )
    return doc_string


for item in data:
    item["documentation"] = preprocess_documentation(item["documentation"])

print(json.dumps(data, indent=2))
for item in data:
    item["documentation"] = item["documentation"].replace("\n", ". ")
with open("preprocessed_cobol_data.json", "w") as f:
    json.dump(data, f, indent=2)

print("Preprocessed data saved to preprocessed_cobol_data.json")


def tokenizer_input(data, max_length=1512):
    tokenized_data = []
    for item in data:
        code_tokens = tokenizer(
            item["cobol_code"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        doc_tokens = tokenizer(
            item["documentation"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        padded_code_tokens = {k: v[:max_length] for k, v in code_tokens.items()}
        padded_doc_tokens = {k: v[:max_length] for k, v in doc_tokens.items()}

        tokenized_data.append(
            {
                "input_ids": padded_code_tokens["input_ids"],
                "attention_mask": padded_code_tokens["attention_mask"],
                "labels": padded_doc_tokens["input_ids"],
            }
        )
    return tokenized_data


tokenized_data = tokenizer_input(data)
print(tokenized_data)
model_name = "t5-small"

model = T5ForConditionalGeneration.from_pretrained(model_name)
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

trainer.train()
model.save_pretrained("outputs")