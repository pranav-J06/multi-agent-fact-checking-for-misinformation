"""
Training module for the Verdict Aggregator MLP Head.
Trains on 5,000 FEVER claim feature representations using CrossEntropyLoss and Adam optimizer.
Saves model checkpoint and standard scaler to models/ directory.
"""

import os
import copy
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

import sys
import argparse
from src.config import (
    AGGREGATOR_MODEL_PATH,
    AGGREGATOR_SCALER_PATH,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
    TRAIN_WEIGHT_DECAY,
    CLASS_NAMES,
    DATA_DIR
)
from src.aggregator.model import VerdictAggregatorMLP
from data.data_loader import load_training_features, load_empirical_features

def train_aggregator(feature_file: Path = None, n_splits: int = 5):
    is_empirical = feature_file is not None and "empirical" in str(feature_file)
    if is_empirical:
        X, y = load_empirical_features(feature_file)
        src_name = feature_file.name
    else:
        X, y = load_training_features(feature_file=feature_file)
        src_name = feature_file.name if feature_file else "simulated_features_6k.npz"

    print(f"[Trainer] Loaded {len(X)} feature representations from {src_name}...")

    # If running on small empirical set, execute 5-Fold Stratified Cross-Validation
    if is_empirical and len(X) <= 100:
        print(f"\n[Trainer] Running {n_splits}-Fold Stratified Cross-Validation across {len(X)} empirical vectors...")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_accuracies = []
        cv_f1s = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            fold_scaler = StandardScaler()
            X_tr_s = fold_scaler.fit_transform(X_tr)
            X_te_s = fold_scaler.transform(X_te)

            tr_ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
            batch_size = min(TRAIN_BATCH_SIZE, max(4, len(tr_ds) // 2))
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

            fold_model = VerdictAggregatorMLP()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(fold_model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)

            for epoch in range(1, TRAIN_EPOCHS + 1):
                fold_model.train()
                for bx, by in tr_loader:
                    optimizer.zero_grad()
                    logits = fold_model.get_logits(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()

            fold_model.eval()
            with torch.no_grad():
                te_x = torch.tensor(X_te_s, dtype=torch.float32)
                te_logits = fold_model.get_logits(te_x)
                te_preds = torch.argmax(te_logits, dim=-1).numpy()

            f_acc = accuracy_score(y_te, te_preds)
            f_f1 = f1_score(y_te, te_preds, average="macro")
            cv_accuracies.append(f_acc)
            cv_f1s.append(f_f1)
            print(f"  Fold {fold}/{n_splits} | Test Acc: {f_acc:.4f} | Macro F1: {f_f1:.4f}")

        mean_acc = np.mean(cv_accuracies)
        std_acc = np.std(cv_accuracies)
        mean_f1 = np.mean(cv_f1s)
        std_f1 = np.std(cv_f1s)
        print("\n" + "=" * 55)
        print(f"5-Fold Stratified CV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"5-Fold Stratified CV Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
        print("=" * 55)

        # Fit final production model and scaler on the complete empirical set
        print("\n[Trainer] Fitting final production model and scaler on all empirical samples...")
        final_scaler = StandardScaler()
        X_all_scaled = final_scaler.fit_transform(X)
        final_ds = TensorDataset(torch.tensor(X_all_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        final_loader = DataLoader(final_ds, batch_size=min(TRAIN_BATCH_SIZE, max(4, len(final_ds) // 2)), shuffle=True)

        final_model = VerdictAggregatorMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)

        for epoch in range(1, TRAIN_EPOCHS + 1):
            final_model.train()
            for bx, by in final_loader:
                optimizer.zero_grad()
                logits = final_model.get_logits(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        AGGREGATOR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), AGGREGATOR_MODEL_PATH)
        with open(AGGREGATOR_SCALER_PATH, "wb") as f:
            pickle.dump(final_scaler, f)

        print(f"[Trainer] Saved empirical model weights to {AGGREGATOR_MODEL_PATH}")
        print(f"[Trainer] Saved empirical feature scaler to {AGGREGATOR_SCALER_PATH}")
        return final_model, final_scaler

    # Standard split for large datasets (e.g. simulated 6,000 vectors)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    batch_size = min(TRAIN_BATCH_SIZE, max(4, len(train_dataset) // 2)) if len(train_dataset) < TRAIN_BATCH_SIZE else TRAIN_BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = VerdictAggregatorMLP()
    param_count = model.count_parameters()
    print(f"[Trainer] Model initialized with {param_count} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)

    best_val_acc = 0.0
    best_weights = None

    print(f"[Trainer] Training for {TRAIN_EPOCHS} epochs...")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model.get_logits(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val_scaled, dtype=torch.float32)
            val_logits = model.get_logits(val_x)
            val_preds = torch.argmax(val_logits, dim=-1).numpy()
            val_acc = accuracy_score(y_val, val_preds)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == TRAIN_EPOCHS:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch:02d}/{TRAIN_EPOCHS:02d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test_scaled, dtype=torch.float32)
        test_logits = model.get_logits(test_x)
        test_preds = torch.argmax(test_logits, dim=-1).numpy()

    test_acc = accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    print("\n" + "="*50)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")
    print("="*50)
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES, digits=4))

    AGGREGATOR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), AGGREGATOR_MODEL_PATH)
    with open(AGGREGATOR_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[Trainer] Saved model weights to {AGGREGATOR_MODEL_PATH}")
    print(f"[Trainer] Saved feature scaler to {AGGREGATOR_SCALER_PATH}")
    return model, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Verdict Aggregator MLP Head")
    parser.add_argument("--empirical", action="store_true", help="Train on empirical features from real pipeline runs")
    parser.add_argument("--feature-file", type=str, default=None, help="Explicit path to feature .npz file")
    args = parser.parse_args()

    feat_path = None
    if args.empirical:
        feat_path = DATA_DIR / "empirical_features.npz"
    elif args.feature_file:
        feat_path = Path(args.feature_file)

    train_aggregator(feature_file=feat_path)
