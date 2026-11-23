import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConstructiveMultiFloorNet


def load_real_dataset():
  csv_path = None
  for root, _, files in os.walk('/kaggle/input'):
    if 'train.csv' in files:
      csv_path = os.path.join(root, 'train.csv')
      break

  if csv_path is not None:
    df = pd.read_csv(csv_path)
    if 'Transported' in df.columns:
      y = df['Transported'].astype(np.float32).values
      drop_c = ['PassengerId', 'Name', 'Cabin', 'Transported']
      df = df.drop(columns=[c for c in drop_c if c in df.columns]).copy()
      for c in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        df[c] = (
            df[c]
            .fillna('MISSING')
            .astype('category')
            .cat.codes.astype(np.float32)
        )
      for c in df.columns:
        df[c] = df[c].fillna(df[c].median())
      return 'Spaceship Titanic', df.values.astype(np.float32), y

  d = load_breast_cancer()
  return (
      'Breast Cancer (Real)',
      d.data[:, :12].astype(np.float32),
      d.target.astype(np.float32),
  )


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  ds_name, X_raw, y_raw = load_real_dataset()

  X_temp, X_te_raw, y_temp, y_te = train_test_split(
      X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=42
  )
  X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
      X_temp, y_temp, test_size=0.30, stratify=y_temp, random_state=42
  )

  scaler = StandardScaler()
  X_tr = torch.tensor(scaler.fit_transform(X_tr_raw), device=device)
  X_va = torch.tensor(scaler.transform(X_va_raw), device=device)
  X_te = torch.tensor(scaler.transform(X_te_raw), device=device)

  model = ConstructiveMultiFloorNet(
      n_floors=3, top_k_per_floor=8, beam_candidates=32, seed=42
  ).to(device)
  model.fit(X_tr, y_tr, X_va, y_va)

  probs = model.predict_proba(X_te).cpu().numpy()
  auc = roc_auc_score(y_te, probs)
  acc = accuracy_score(y_te, probs > 0.5) * 100.0

  print('=' * 76)
  print(f'ConstructiveMultiFloorNet ({ds_name}) | Test AUC: {auc:.4f} | Test Acc: {acc:.2f}%')
  print('=' * 76)


if __name__ == '__main__':
  main()
