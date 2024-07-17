import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

print(torch.cuda.is_available())
torch.cuda.empty_cache()
gc.collect()

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left')
#model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")
#tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

data = {
    "inputs": ["What is surname of Ania from Computer Science in Poznan University of Technology?", "How old is Ania Kowalska?", "Do Anna Kowalska have siblings?", "What is studing Ania Kowalska?"],
    "outputs": ["Kowalska", "21", "Michał Kowalski", "Computer Science, but she want to become graphic designer."]
}

df = pd.DataFrame(data)
dataset_custom = Dataset.from_pandas(df)

def preprocess_function(examples):
    inputs = [str(inp) for inp in examples['inputs']]
    outputs = [str(out) for out in examples['outputs']]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset_custom.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    fp16=True,
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
