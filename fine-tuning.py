import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

dataset_hf = load_dataset("nomic-ai/gpt4all-j-prompt-generations", revision="v1.2-jazzy")

model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
sentences = ['This is an example.', 'Another sentence.']

tokenized_batch = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

#print(tokenized_batch)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

data = {
    "inputs": ["Jak ma na nazwisko Jagoda?", "Ile lat ma Jagoda Janowska?", "Kto jest bratem Jagody Janowskiej?", "Co studiuje Jagoda Janowska?"],
    "outputs": ["Janowska", "21", "Szymon Janowski", "Informatykę, ale stara się zmienić studia na weterynarię."]
}

df = pd.DataFrame(data)
dataset_custom = Dataset.from_pandas(df)

dataset = concatenate_datasets([dataset_hf['train'], dataset_custom])

#inputs = ["Jak ma na nazwisko Ania?", "Ile lat ma Jagoda Janowska", "Kto jest bratem Ani Jankowskiej?", "Co studiuje Anna Jankowska?"]
#outputs = ["Jankowska", "21", "Michał Jankowski", "Informatykę, ale stara się zmienić studia na weterynarię."]

def preprocess_function(examples):
    inputs = [str(inp) for inp in examples['inputs']]
    outputs = [str(out) for out in examples['outputs']]

    #print("Inputs before tokenization:", inputs)
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    #print("Tokenized inputs:", model_inputs)

    #print("Outputs before tokenization:", outputs)
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    #print("Tokenized labels:", labels)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

trainer.train()

trainer.save_model("trained_model_1")
tokenizer.save_pretrained("trained_model_1")
