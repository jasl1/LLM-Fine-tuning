test_dataset = tokenized_datasets["test"]
test_results = trainer.predict(test_dataset)

for i, prediction in enumerate(test_results.predictions):
    print(f"Article: {test_dataset['article'][i]}")
    print(f"Summary: {tokenizer.decode(prediction, skip_special_tokens=True)}")
