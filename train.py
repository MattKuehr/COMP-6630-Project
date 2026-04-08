import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import get_data
from models import SentimentRNN

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        text = batch["input_ids"]
        labels = batch["label"].unsqueeze(1)
        
        predictions = model(text)
        loss = criterion(predictions, labels)
        
        # Calculate accuracy
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == labels).float()
        acc = correct.sum() / len(correct)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text = batch["input_ids"]
            labels = batch["label"].unsqueeze(1)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            rounded_preds = torch.round(predictions)
            correct = (rounded_preds == labels).float()
            acc = correct.sum() / len(correct)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def run_experiment(model_type, train_loader, test_loader, vocab_size):
    print(f"\n--- Training {model_type.upper()} ---")
    
    # Hyperparameters
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    EPOCHS = 3 # Kept low for quick execution
    
    model = SentimentRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                         N_LAYERS, BIDIRECTIONAL, DROPOUT, model_type=model_type)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, test_loader, criterion)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Valid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')
        
    return valid_acc

if __name__ == "__main__":
    train_loader, test_loader, vocab_size = get_data()
    
    results = {}
    for m_type in ["rnn", "gru", "lstm"]:
        acc = run_experiment(m_type, train_loader, test_loader, vocab_size)
        results[m_type] = acc
        
    print("\nFinal Comparison:")
    for m_type, acc in results.items():
        print(f"{m_type.upper()}: {acc*100:.2f}%")
