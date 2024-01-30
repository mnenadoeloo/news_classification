from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, DataCollatorWithPadding

news = load_dataset("ag_news")

checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint, do_lower_case=True)

def tokenize_function(example):
  return tokenizer(example["text"], truncation=True)

tokenized_datasets = news.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns("text")
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["test"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}
