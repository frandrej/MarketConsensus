import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import optuna
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = r"C:\Users\diefr\OneDrive\Lernsachen\MarketConsensus\data\processed"
FEATURES_PATH = f"{DATA_DIR}\\features_dataset.csv"
MODEL_OUTPUT_DIR = DATA_DIR
RESULTS_OUTPUT_PATH = f"{MODEL_OUTPUT_DIR}\\model_results_fixed.txt"

RANDOM_STATE = 42
N_TRIALS = 50

CLASS_WEIGHTS = {
    0: 2.0,
    1: 0.5,
    2: 20.0
}

# ============================================================================
# DATA PREPARATION - NO DATA LEAKAGE
# ============================================================================

def load_data(filepath):
    """Load data"""
    print("=" * 80)
    print("IMPROVED MODEL - FIXED DATA LEAKAGE")
    print("=" * 80)
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\nLoaded {len(df)} records")
    print(f"\nClass Distribution:")
    print(df['target_label'].value_counts())
    
    return df

def prepare_features_no_leakage(df, train_end, val_end):
    """
    Prepare features WITHOUT data leakage
    - Split FIRST
    - Compute imputation/scaling from training data ONLY
    - Apply same transformations to val/test
    """
    print("\n" + "=" * 80)
    print("FEATURE PREPARATION (NO LEAKAGE)")
    print("=" * 80)
    
    # Exclude columns
    exclude_cols = [
        'ticker', 'date', 'close',
        'target_label', 'target_numeric',
        'future_vol', 'vol_change', 'vol_change_pct', 'vol_change_ratio',
        'historical_vol'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"\nUsing {len(feature_cols)} features")
    
    X = df[feature_cols].copy()
    y = df['target_numeric'].copy()
    
    # SPLIT FIRST
    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    
    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]
    
    print(f"\nSplit sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # COMPUTE imputation values from TRAINING data only
    print("\nComputing imputation values from training data...")
    imputation_dict = {}
    for col in feature_cols:
        train_median = X_train[col].median()
        if pd.isna(train_median):
            train_median = 0
        imputation_dict[col] = train_median
    
    # APPLY imputation
    for col in feature_cols:
        X_train[col].fillna(imputation_dict[col], inplace=True)
        X_val[col].fillna(imputation_dict[col], inplace=True)
        X_test[col].fillna(imputation_dict[col], inplace=True)
    
    # Handle infinity
    for df_subset in [X_train, X_val, X_test]:
        df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feature_cols:
            df_subset[col].fillna(imputation_dict[col], inplace=True)
    
    # SCALE using training statistics only
    print("Scaling features using training statistics...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),  # FIT on train
        columns=feature_cols,
        index=X_train.index
    )
    
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),  # TRANSFORM val (no fit)
        columns=feature_cols,
        index=X_val.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),  # TRANSFORM test (no fit)
        columns=feature_cols,
        index=X_test.index
    )
    
    print("Features prepared without data leakage ✓")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_cols

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

def objective(trial, X_train, y_train, X_val, y_val, class_weights):
    """Optuna objective"""
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-3, 1.0, log=True),
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'force_row_wise': True
    }
    
    # Sample weights
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)
        ]
    )
    
    y_pred = np.argmax(model.predict(X_val), axis=1)
    return f1_score(y_val, y_pred, average='weighted')

def optimize_hyperparameters(X_train, y_train, X_val, y_val, class_weights, n_trials):
    """Run Optuna optimization"""
    print("\n" + "=" * 80)
    print(f"HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
    print("=" * 80)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, class_weights),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\nBest F1: {study.best_trial.value:.4f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
    
    best_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'force_row_wise': True,
        **study.best_trial.params
    }
    
    return best_params, study

# ============================================================================
# TRAINING
# ============================================================================

def train_final_model(X_train, y_train, X_val, y_val, params, class_weights):
    """Train final model"""
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print(f"\nFinal model: {model.num_trees()} trees")
    return model

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def find_optimal_thresholds(model, X_val, y_val):
    """Find optimal thresholds"""
    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    y_pred_proba = model.predict(X_val)
    
    best_thresholds = [0.33, 0.33, 0.33]
    best_f1 = 0
    
    for t0 in np.arange(0.1, 0.6, 0.05):
        for t1 in np.arange(0.1, 0.6, 0.05):
            for t2 in np.arange(0.1, 0.6, 0.05):
                y_pred = []
                for proba in y_pred_proba:
                    if proba[0] > t0:
                        y_pred.append(0)
                    elif proba[1] > t1:
                        y_pred.append(1)
                    elif proba[2] > t2:
                        y_pred.append(2)
                    else:
                        y_pred.append(np.argmax(proba))
                
                y_pred = np.array(y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = [t0, t1, t2]
    
    print(f"\nOptimal thresholds: {best_thresholds}")
    print(f"Validation F1: {best_f1:.4f}")
    
    return best_thresholds

def apply_thresholds(proba, thresholds):
    """Apply custom thresholds"""
    y_pred = []
    for p in proba:
        if p[0] > thresholds[0]:
            y_pred.append(0)
        elif p[1] > thresholds[1]:
            y_pred.append(1)
        elif p[2] > thresholds[2]:
            y_pred.append(2)
        else:
            y_pred.append(np.argmax(p))
    return np.array(y_pred)

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, X_test, y_test, feature_cols, thresholds, output_path):
    """Evaluate model"""
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    y_pred_proba = model.predict(X_test)
    y_pred = apply_thresholds(y_pred_proba, thresholds)
    y_pred_std = np.argmax(y_pred_proba, axis=1)
    
    acc_std = accuracy_score(y_test, y_pred_std)
    acc_thresh = accuracy_score(y_test, y_pred)
    f1_thresh = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\nStandard: Accuracy={acc_std:.4f}")
    print(f"Threshold: Accuracy={acc_thresh:.4f}, F1={f1_thresh:.4f}")
    
    with open(output_path, 'w') as f:
        f.write("FIXED MODEL - NO DATA LEAKAGE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Standard Accuracy: {acc_std:.4f}\n")
        f.write(f"Threshold Accuracy: {acc_thresh:.4f}\n")
        f.write(f"Weighted F1: {f1_thresh:.4f}\n\n")
        
        report = classification_report(
            y_test, y_pred,
            labels=[0, 1, 2],
            target_names=['Decrease', 'Stay Same', 'Increase'],
            zero_division=0
        )
        print(f"\n{report}")
        f.write(report + "\n")
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        
        fi = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        f.write("\nTop 20 Features:\n")
        f.write(fi.head(20).to_string(index=False))
        
        fi.to_csv(f"{MODEL_OUTPUT_DIR}\\feature_importance_fixed.csv", index=False)
    
    print(f"\nResults saved: {output_path}")
    return fi

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load
    df = load_data(FEATURES_PATH)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Define splits (70/20/10)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.90)
    
    print(f"\nSplit: Train={train_end}, Val={val_end-train_end}, Test={n-val_end}")
    
    # Prepare features without leakage
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_features_no_leakage(
        df, train_end, val_end
    )
    
    # Optimize
    best_params, study = optimize_hyperparameters(
        X_train, y_train, X_val, y_val, CLASS_WEIGHTS, N_TRIALS
    )
    
    # Train
    model = train_final_model(X_train, y_train, X_val, y_val, best_params, CLASS_WEIGHTS)
    
    # Threshold optimization
    thresholds = find_optimal_thresholds(model, X_val, y_val)
    
    # Evaluate
    fi = evaluate(model, X_test, y_test, feature_cols, thresholds, RESULTS_OUTPUT_PATH)
    
    # Save
    model.save_model(f"{MODEL_OUTPUT_DIR}\\model_fixed.txt")
    pd.DataFrame({
        'class': ['decrease', 'stay_same', 'increase'],
        'threshold': thresholds
    }).to_csv(f"{MODEL_OUTPUT_DIR}\\thresholds_fixed.csv", index=False)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - NO DATA LEAKAGE!")
    print("=" * 80)
    
    return model, fi, thresholds

if __name__ == "__main__":
    model, fi, thresholds = main()