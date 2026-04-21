# Author: Matthew Sawyer
# Author Email: mss0096@auburn.edu

import os
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from torch.utils.data import Dataset, DataLoader


class SarcasmDataset(Dataset):
    """
    A custom PyTorch Dataset for sarcasm detection.

    Attributes:
        ds (datasets.Dataset): The dataset split containing headlines and labels.
        tokenizer (Tokenizer): The tokenizer used to encode the headlines.
    """
    def __init__(self, ds_split, tokenizer):
        """
        Initializes the SarcasmDataset.

        Args:
            ds_split (datasets.Dataset): The specific split of the dataset (e.g., train, test, val).
            tokenizer (Tokenizer): The tokenizer to be used for encoding headlines.
        """
        self.ds = ds_split
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing 'input_ids' (encoded headline) and 'label' (sarcastic or not).
        """
        item = self.ds[idx]
        headline = item["headline"]
        label = item["is_sarcastic"]

        encoding = self.tokenizer.encode(headline)
        ids = encoding.ids
        
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float)
        }


def collate_fn(batch):
    """
    Collates a batch of samples into a single batch of tensors, padding the sequences to the same length.

    Args:
        batch (list): A list of dictionaries, each containing 'input_ids' and 'label'.

    Returns:
        dict: A dictionary containing 'input_ids' (padded), 'label', and 'lengths' (original sequence lengths).
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["label"] for item in batch]
    lengths = torch.tensor([len(ids) for ids in input_ids])
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    
    return {
        "input_ids": padded_input_ids,
        "label": labels,
        "lengths": lengths
    }


def get_data(tokenizer_type="bpe", batch_size=32):
    """
    Loads the sarcasm dataset, trains or loads the specified tokenizer, and prepares DataLoaders.

    Args:
        tokenizer_type (str, optional): The type of tokenizer to use ('bpe', 'word', 'wordpiece', or 'unigram'). Defaults to "bpe".
        batch_size (int, optional): The batch size for DataLoaders. Defaults to 32.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader, vocab_size).

    Raises:
        ValueError: If an unknown tokenizer_type is provided.
    """
    dataset = load_dataset("raquiba/Sarcasm_News_Headline")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer_path = os.path.join(tokenizer_dir, f"tokenizer_{tokenizer_type}.json")
    
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)
    
    if not os.path.exists(tokenizer_path):
        print(f"Training {tokenizer_type} tokenizer...")
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
        if tokenizer_type == "bpe":
            tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
            trainer = trainers.BpeTrainer(special_tokens=special_tokens)
        elif tokenizer_type == "word":
            tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
        elif tokenizer_type == "wordpiece":
            tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            trainer = trainers.WordPieceTrainer(special_tokens=special_tokens)
        elif tokenizer_type == "unigram":
            tokenizer = Tokenizer(models.Unigram())
            tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
            trainer = trainers.UnigramTrainer(special_tokens=special_tokens, unk_token="[UNK]")
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        def batch_iterator():
            for i in range(0, len(dataset["train"]), 1000):
                yield dataset["train"][i : i + 1000]["headline"]
        
        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)

    train_ds = SarcasmDataset(dataset["train"], tokenizer)
    test_val_ds = dataset["test"].train_test_split(test_size=0.5, seed=42)
    val_ds = SarcasmDataset(test_val_ds["train"], tokenizer)
    test_ds = SarcasmDataset(test_val_ds["test"], tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, tokenizer.get_vocab_size()
