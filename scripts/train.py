# Author: Matt Kuehr
# Email: mck0063@auburn.edu

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from scripts.preprocess import get_data
from scripts.models import SentimentRNN


def train(model, iterator, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to be trained.
        iterator (DataLoader): The DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for updates.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU or GPU) to run the training on.

    Returns:
        tuple: A tuple containing (average_epoch_loss, average_epoch_accuracy).
    """
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        text = batch["input_ids"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)
        lengths = batch["lengths"]
        
        predictions = model(text, lengths)
        loss = criterion(predictions, labels)

        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == labels).float()
        acc = correct.sum() / len(correct)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Evaluates the model on a given dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        iterator (DataLoader): The DataLoader providing the evaluation data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU or GPU) to run the evaluation on.

    Returns:
        tuple: A tuple containing (average_epoch_loss, average_epoch_accuracy).
    """
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text = batch["input_ids"].to(device)
            labels = batch["label"].unsqueeze(1).to(device)
            lengths = batch["lengths"]
            
            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            
            rounded_preds = torch.round(predictions)
            correct = (rounded_preds == labels).float()
            acc = correct.sum() / len(correct)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def run_experiment(model_type, tokenizer_type, train_loader, val_loader, test_loader, vocab_size, device):
    """
    Runs a training and evaluation experiment for a specific model and tokenizer combination.

    Args:
        model_type (str): The type of model to use ('rnn', 'gru', or 'lstm').
        tokenizer_type (str): The type of tokenizer being used.
        train_loader (DataLoader): The DataLoader for training data.
        val_loader (DataLoader): The DataLoader for validation data.
        test_loader (DataLoader): The DataLoader for test data.
        vocab_size (int): The size of the vocabulary.
        device (torch.device): The device to run the experiment on.

    Returns:
        tuple: A tuple containing the best metrics recorded during the experiment:
               (train_loss, train_acc, valid_loss, valid_acc).
    """
    print(f"\n--- Training {model_type.upper()} with {tokenizer_type.upper()} on {device} ---")
    
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    EPOCHS = 3
    
    model = SentimentRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                         N_LAYERS, BIDIRECTIONAL, DROPOUT, model_type=model_type)
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    criterion = criterion.to(device)
    
    best_valid_acc = 0
    best_metrics = (0, 0, 0, 0)
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_metrics = (train_loss, train_acc, valid_loss, valid_acc)
            
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Final Test Acc: {test_acc*100:.2f}%')
        
    return best_metrics


def print_summary_table(results):
    """
    Prints a formatted summary table of the experiment results.

    Args:
        results (dict): A nested dictionary containing experiment results. 
                        Format: {tokenizer_type: {model_type: (metrics_tuple)}}.
    """
    print("\n" + "="*122)
    print(f"{'Tokenizer':<12} | {'RNN (Train L/A | Val L/A)':<34} | {'GRU (Train L/A | Val L/A)':<34} | {'LSTM (Train L/A | Val L/A)':<34}")
    print("-" * 122)
    for t_type, models in results.items():
        row = f"{t_type.upper():<12}"
        for m_type in ["rnn", "gru", "lstm"]:
            tl, ta, vl, va = models.get(m_type, (0, 0, 0, 0))
            metrics_str = f"{tl:.3f}/{ta*100:5.2f}% | {vl:.3f}/{va*100:5.2f}%"
            row += f" | {metrics_str:<34}"
        print(row)
    print("="*122)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer_types = ["bpe", "word", "wordpiece", "unigram"]
    model_types = ["rnn", "gru", "lstm"]
    
    all_results = {}
    for t_type in tokenizer_types:
        train_loader, val_loader, test_loader, vocab_size = get_data(tokenizer_type=t_type)
        all_results[t_type] = {}
        for m_type in model_types:
            metrics = run_experiment(m_type, t_type, train_loader, val_loader, test_loader, vocab_size, device)
            all_results[t_type][m_type] = metrics
            
    print_summary_table(all_results)
