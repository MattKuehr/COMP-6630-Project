import os
import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader

class SarcasmDataset(Dataset):
    def __init__(self, ds_split, tokenizer, max_length=50):
        self.ds = ds_split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        headline = item["headline"]
        label = item["is_sarcastic"]

        encoding = self.tokenizer.encode(headline)
        ids = encoding.ids
        
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        else:
            ids = ids + [0] * (self.max_length - len(ids))

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)
        }

def get_data(batch_size=32, max_len=50):
    dataset = load_dataset("raquiba/Sarcasm_News_Headline")
    
    if not os.path.exists("tokenizer.json"):
        print("Training tokenizer...")
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        
        def batch_iterator():
            for i in range(0, len(dataset["train"]), 1000):
                yield dataset["train"][i : i + 1000]["headline"]
        
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save("tokenizer.json")
    else:
        tokenizer = Tokenizer.from_file("tokenizer.json")

    train_ds = SarcasmDataset(dataset["train"], tokenizer, max_length=max_len)
    test_ds = SarcasmDataset(dataset["test"], tokenizer, max_length=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, tokenizer.get_vocab_size()
