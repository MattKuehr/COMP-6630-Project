# Sarcasm Detection in News Headlines

This project implements and evaluates deep learning models for sarcasm detection in news headlines. It utilizes the `raquiba/Sarcasm_News_Headline` dataset and explores various recurrent neural network architectures combined with different tokenization strategies.

## Project Scope

The goal of this project is to benchmark the performance of different model architectures and tokenization methods on the task of binary sarcasm classification.

### Key Features:
- **Model Architectures:** Supports Vanilla RNN, GRU, and LSTM.
- **Tokenization Strategies:** Includes Word-level, BPE (Byte Pair Encoding), WordPiece, and Unigram tokenizers.
- **Comparative Analysis:** A unified script to run experiments across all combinations of models and tokenizers.
- **OOD Evaluation:** Tools to evaluate model robustness on Out-of-Distribution (OOD) data.
- **Reproducibility:** Dedicated notebooks for re-training the best model and performing detailed evaluations.

## Directory Structure

```text
├── checkpoints/           # Saved model weights and training states
├── notebooks/
│   ├── best_model_training.ipynb  # Re-training the top-performing model
│   └── ood_evaluation.ipynb      # Evaluating model on OOD datasets
├── ood_data/              # JSON files containing OOD evaluation samples
├── scripts/
│   ├── models.py          # SentimentRNN model definition (RNN/GRU/LSTM)
│   ├── preprocess.py      # Data loading and tokenizer management
│   └── train.py           # Main experiment execution script
├── tokenizer/             # Saved tokenizer configurations
├── requirements.txt       # Project dependencies
└── requirements_cpu.txt   # CPU-optimized dependencies
```

## Installation

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   - For GPU support: `pip install -r requirements.txt`
   - For CPU-only: `pip install -r requirements_cpu.txt`

## Usage

### Running Comparative Experiments
The `scripts/train.py` script automatically trains and evaluates all combinations of models (RNN, GRU, LSTM) and tokenizers (BPE, Word, WordPiece, Unigram).

```bash
python scripts/train.py
```
This will print a summary table of the best training and validation metrics for each combination.

### Re-training the Best Model
For a more intensive training session of the best-performing model (LSTM with WordLevel tokenizer), use the provided notebook:
- **File:** `notebooks/best_model_training.ipynb`
- **Details:** This notebook runs for 25 epochs, saves checkpoints for every epoch, and keeps the `best_model.pt` based on validation loss.

### Out-of-Distribution (OOD) Evaluation
To test how well the model generalizes to data it hasn't seen during training or standard testing:
- **File:** `notebooks/ood_evaluation.ipynb`
- **Data:** Uses samples from `ood_data/train1.json` and `ood_data/train2.json`.
- **Functionality:** Loads saved checkpoints and reports accuracy on the OOD sets.

## Model Details

The `SentimentRNN` class in `scripts/models.py` is the core architecture. It consists of:
1. **Embedding Layer:** Converts input tokens to dense vectors.
2. **Recurrent Layer:** A multi-layer, bidirectional RNN, GRU, or LSTM.
3. **Dropout:** Applied for regularization.
4. **Fully Connected Layer:** Maps the final hidden state to a single output.
5. **Sigmoid Activation:** Outputs the probability of the headline being sarcastic.

## Data Source
The project uses the `raquiba/Sarcasm_News_Headline` dataset from Hugging Face, which is automatically downloaded via the `datasets` library in `scripts/preprocess.py`.
