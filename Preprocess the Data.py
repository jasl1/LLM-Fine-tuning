from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")

def preprocess_data(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, return_tensors="pt")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=128, truncation=True, return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True)
