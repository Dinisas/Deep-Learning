import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils import (
    load_rnacompete_data,
    masked_mse_loss,
    masked_spearman_correlation,
    configure_seed,
    plot
)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

configure_seed(42)

class vanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(vanillaRNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers

        #RNN Layer
        self.rnn=nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0,
            nonlinearity='relu')

        #Fully connected layers
        self.fc1=nn.Linear(in_features=hidden_size, out_features=32)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.fc2=nn.Linear(in_features=32, out_features=1)

    def forward(self,x):
        # RNN forward pass
        output, h_n = self.rnn(x)
        last_hidden = h_n[-1]

        # Fully connected layers
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_attention=False, num_attention_heads=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)

        # Optional Attention
        if use_attention:
            self.attention = AttentionPooling(hidden_size, num_heads=num_attention_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        # LSTM forward pass
        output, (h_n, c_n) = self.lstm(x)

        # Use attention or last hidden state
        if self.use_attention:
            pooled = self.attention(output)
        else:
            pooled = h_n[-1]

        # Fully connected layers
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_attention=False, num_attention_heads=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        # GRU Layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0)

        # Optional Attention
        if use_attention:
            self.attention = AttentionPooling(hidden_size, num_heads=num_attention_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        # GRU forward pass
        output, h_n = self.gru(x)

        # Use attention or last hidden state
        if self.use_attention:
            pooled = self.attention(output)
        else:
            pooled = h_n[-1]

        # Fully connected layers
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentionPooling(nn.Module):
    """
    Attention pooling layer that computes weighted sum of hidden states.
    """
    def __init__(self, hidden_size, num_heads=1):
        super(AttentionPooling, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        if num_heads == 1:
            self.attention_weights = nn.Linear(hidden_size, 1)
        else:
            self.attention_weights = nn.Linear(hidden_size, num_heads)
            self.output_projection = nn.Linear(hidden_size * num_heads, hidden_size)

    def forward(self, hidden_states):
        scores = self.attention_weights(hidden_states)

        if self.num_heads == 1:
            scores = scores.squeeze(-1)
            attention_weights = nn.functional.softmax(scores, dim=1)
            attended = torch.bmm(
                attention_weights.unsqueeze(1),
                hidden_states
            ).squeeze(1)
        else:
            attention_weights = nn.functional.softmax(scores, dim=1)
            attended_heads = []
            for head in range(self.num_heads):
                head_weights = attention_weights[:, :, head]
                head_attended = torch.bmm(
                    head_weights.unsqueeze(1),
                    hidden_states
                ).squeeze(1)
                attended_heads.append(head_attended)
            attended = torch.cat(attended_heads, dim=1)
            attended = self.output_projection(attended)

        return attended

#Training Function
def train_epoch(model,loader, criterion,optimizer):
    model.train()
    total_loss = 0.0
    for sequences, targets, masks in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        predictions= model(sequences)
        loss=criterion(predictions, targets, masks)
        loss.backward()

        #gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

#Evaluation Function
def evaluate(model,loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for sequences, targets, masks in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, targets, masks)
            total_loss += loss.item()

            # Store for correlation calculation
            all_preds.append(predictions)
            all_targets.append(targets)
            all_masks.append(masks)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate Spearman correlation
    correlation = masked_spearman_correlation(all_preds, all_targets, all_masks)
    avg_loss = total_loss / len(loader)

    return avg_loss, correlation.item()

def grid_search(args):
    """Perform grid search over hyperparameters."""

    # Define hyperparameter grid
    param_grid = {
        'model_type': ['lstm', 'gru'],
        'hidden_size': [64, 128],
        'num_layers': [2, 3],
        'dropout': [0.2, 0.3],
        'learning_rate': [0.001, 0.0005],
    }

    # Generate all combinations
    import itertools
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Testing {len(combinations)} configurations...\n")

    all_results = []

    for idx, params in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] {params['model_type'].upper()} h={params['hidden_size']} l={params['num_layers']} d={params['dropout']} lr={params['learning_rate']}")

        # Call your existing main with these params
        result = main(
            model_type=params['model_type'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            batch_size=args.batch_size,
            epochs=args.epochs
        )

        result['params'] = params
        all_results.append(result)
        print(f"  → Best Val Corr: {result['best_val_corr']:.4f}\n")

    # Sort and show best
    sorted_results = sorted(all_results, key=lambda x: x['best_val_corr'], reverse=True)

    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS")
    print("="*60)
    for i, r in enumerate(sorted_results[:5], 1):
        p = r['params']
        print(f"{i}. {p['model_type'].upper()} h={p['hidden_size']} l={p['num_layers']} d={p['dropout']} lr={p['learning_rate']} → {r['best_val_corr']:.4f}")

    return sorted_results

def main(model_type='lstm', hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001, batch_size=64, epochs=50, use_attention=False, num_attention_heads=1):
    #Load data
    train_dataset = load_rnacompete_data('RBFOX1', split='train')
    val_dataset = load_rnacompete_data('RBFOX1', split='val')
    test_dataset = load_rnacompete_data('RBFOX1', split='test')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model_type == 'vanilla':
        model = vanillaRNN(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

    elif model_type == 'lstm':
        model = LSTM(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads
        ).to(device)

    else:
        model = GRU(
            input_size=4,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads
        ).to(device)

    criterion=masked_mse_loss
    optimizer=optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nStarting training...\n")
    train_losses = []
    val_losses = []
    val_correlations = []

    best_val_corr = -1.0
    best_epoch = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer)

        # Evaluate
        val_loss, val_corr = evaluate(model, val_loader, criterion)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_correlations.append(val_corr)

        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_epoch = epoch
            attn_suffix = f'_attn{num_attention_heads}h' if use_attention else ''
            torch.save(model.state_dict(), f'best_{model_type}{attn_suffix}_model.pth')

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Corr: {val_corr:.4f}")

    print(f"\nBest validation correlation: {best_val_corr:.4f} at epoch {best_epoch+1}")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'best_{model_type}{attn_suffix}_model.pth'))
    test_loss, test_corr = evaluate(model, test_loader, criterion)

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Spearman Correlation: {test_corr:.4f}")
    print(f"{'='*60}\n")

    # Plot results
    epochs_range = list(range(1, epochs + 1))

    # Plot losses
    plot(
        epochs_range,
        {'Train Loss': train_losses, 'Val Loss': val_losses},
        filename=f'{model_type}{attn_suffix}_losses.pdf'
    )
    print(f"Saved: {model_type}_losses.pdf")

    # Plot correlation
    plot(
        epochs_range,
        {'Val Spearman Correlation': val_correlations},
        filename=f'{model_type}{attn_suffix}_correlation.pdf'
    )
    print(f"Saved: {model_type}_correlation.pdf")

    return {
        'model_type': model_type,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_correlations': val_correlations,
        'test_loss': test_loss,
        'test_correlation': test_corr,
        'best_val_corr': best_val_corr,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RNN-based RNA Binding Protein Interaction Prediction (Q2.1 & Q2.2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model architecture arguments
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=['vanilla', 'lstm', 'gru'],
                        help='Type of RNN model to use')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='Hidden size for RNN layers')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')

    # Attention arguments (for Q2.2)
    parser.add_argument('--use_attention', action='store_true',
                        help='Whether to use attention pooling (Q2.2)')
    parser.add_argument('--num_attention_heads', type=int, default=1,
                        choices=[1, 2, 4],
                        help='Number of attention heads')

    # Training arguments
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.001,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Mode selection
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'grid_search', 'attention_comparison'],
                        help='Running mode: single model, grid search, or attention comparison (Q2.2)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Set random seed
    configure_seed(args.seed)

    if args.mode == 'single':
        # Train a single model with specified arguments
        print("\n" + "="*60)
        print(f"Training {args.model_type.upper()} model")
        print(f"Hidden size: {args.hidden_size}, Layers: {args.num_layers}")
        print(f"Dropout: {args.dropout}, LR: {args.learning_rate}")
        print(f"Attention: {args.use_attention}, Heads: {args.num_attention_heads}")
        print("="*60)

        result = main(
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_attention=args.use_attention,
            num_attention_heads=args.num_attention_heads
        )

    elif args.mode == 'grid_search':
        # Run grid search (Q2.1)
        print("\n" + "="*60)
        print("GRID SEARCH MODE (Q2.1)")
        print("="*60)
        all_results = grid_search(args)

    elif args.mode == 'attention_comparison':
        # Run attention comparison (Q2.2)
        print("\n" + "="*60)
        print("ATTENTION COMPARISON MODE (Q2.2)")
        print(f"Base model: {args.model_type.upper()}")
        print("="*60)

        # BASELINE: WITHOUT attention
        print("\n" + "="*60)
        print(f"BASELINE: {args.model_type.upper()} WITHOUT ATTENTION")
        print("="*60)
        baseline_result = main(
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_attention=False
        )

        # WITH ATTENTION: Try different head counts
        attention_results = {}
        for num_heads in [1, 2, 4]:
            print("\n" + "="*60)
            print(f"{args.model_type.upper()} WITH {num_heads} ATTENTION HEAD(S)")
            print("="*60)
            result = main(
                model_type=args.model_type,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                epochs=args.epochs,
                use_attention=True,
                num_attention_heads=num_heads
            )
            attention_results[num_heads] = result

        # COMPARISON SUMMARY
        print("\n" + "="*60)
        print("COMPARISON: BASELINE vs ATTENTION")
        print("="*60)
        print(f"Baseline: Val={baseline_result['best_val_corr']:.4f}, Test={baseline_result['test_correlation']:.4f}")
        for num_heads, result in attention_results.items():
            print(f"{num_heads} head(s): Val={result['best_val_corr']:.4f}, Test={result['test_correlation']:.4f}")