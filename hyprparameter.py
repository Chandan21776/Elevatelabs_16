"""
============================================================
  Task 16: Hyperparameter Tuning using GridSearchCV
  Dataset: Breast Cancer (sklearn)
  Author: ML Internship Task
============================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

print("=" * 80)
print("  HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("=" * 80)

# ──────────────────────────────────────────────────────────────
# STEP 1: LOAD & PREPROCESS
# ──────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=[str(f) for f in data.feature_names])
y = data.target

print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"  Classes: {list(data.target_names)}")

# ──────────────────────────────────────────────────────────────
# STEP 2: TRAIN/TEST SPLIT
# ──────────────────────────────────────────────────────────────
print("\n[STEP 2] Splitting Data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")

# ──────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ──────────────────────────────────────────────────────────────
print("\n[STEP 3] Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✓ Features normalized")

# ──────────────────────────────────────────────────────────────
# STEP 4: DEFINE PARAMETER GRIDS
# ──────────────────────────────────────────────────────────────
print("\n[STEP 4] Defining Parameter Grids...")

param_grids = {
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=10000, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }
}

for name, cfg in param_grids.items():
    n_combinations = np.prod([len(v) for v in cfg['params'].values()])
    print(f"  {name}: {n_combinations} parameter combinations")

# ──────────────────────────────────────────────────────────────
# STEP 5: BASELINE - DEFAULT MODELS
# ──────────────────────────────────────────────────────────────
print("\n[STEP 5] Training DEFAULT Models (Baseline)...")
print("-" * 80)

default_results = {}
for name, cfg in param_grids.items():
    model = cfg['model']
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    default_results[name] = {
        'accuracy': acc,
        'f1_score': f1,
        'auc_roc': auc,
        'time': train_time
    }
    
    print(f"{name:>20}: Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | Time={train_time:.3f}s")

# ──────────────────────────────────────────────────────────────
# STEP 6: GRIDSEARCHCV - HYPERPARAMETER TUNING
# ──────────────────────────────────────────────────────────────
print("\n[STEP 6] Running GridSearchCV (5-Fold CV)...")
print("-" * 80)

tuned_results = {}
for name, cfg in param_grids.items():
    print(f"\nTuning {name}...")
    start = time.time()
    
    grid_search = GridSearchCV(
        estimator=cfg['model'],
        param_grid=cfg['params'],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    tune_time = time.time() - start
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    tuned_results[name] = {
        'accuracy': acc,
        'f1_score': f1,
        'auc_roc': auc,
        'time': tune_time,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_
    }
    
    print(f"  Best Params: {grid_search.best_params_}")
    print(f"  CV F1 Score: {grid_search.best_score_:.4f}")
    print(f"  Test Metrics: Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
    print(f"  Tuning Time: {tune_time:.2f}s")

# ──────────────────────────────────────────────────────────────
# STEP 7: PERFORMANCE COMPARISON
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("  PERFORMANCE COMPARISON: DEFAULT vs TUNED")
print("=" * 80)

print(f"\n{'Model':<22} {'Version':<10} {'Accuracy':<12} {'F1-Score':<12} {'AUC-ROC':<12} {'Improvement'}")
print("-" * 90)

for name in param_grids.keys():
    default_f1 = default_results[name]['f1_score']
    tuned_f1 = tuned_results[name]['f1_score']
    improvement = ((tuned_f1 - default_f1) / default_f1) * 100
    
    print(f"{name:<22} {'Default':<10} {default_results[name]['accuracy']:.4f}       {default_f1:.4f}       {default_results[name]['auc_roc']:.4f}")
    print(f"{name:<22} {'Tuned':<10} {tuned_results[name]['accuracy']:.4f}       {tuned_f1:.4f}       {tuned_results[name]['auc_roc']:.4f}       {improvement:+.2f}%")
    print()

print("=" * 80)
print("  HYPERPARAMETER TUNING COMPLETE")
print("=" * 80)

# ──────────────────────────────────────────────────────────────
# BEST PARAMETERS SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n📋 BEST PARAMETERS FOUND:")
print("-" * 80)
for name, result in tuned_results.items():
    print(f"\n{name}:")
    for param, value in result['best_params'].items():
        print(f"  • {param}: {value}")