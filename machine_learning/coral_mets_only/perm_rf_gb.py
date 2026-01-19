import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import functools

# 1. Load Main Data
coral_df = pd.read_csv("/work/hs325/coral/data/corals.csv")
met_df = pd.read_csv("/work/hs325/coral/data/metabolites.csv")

# Preprocessing
coral_df['scleractinia'] = np.where(coral_df['host_order'] == 'Scleractinia', 1, 0)
met_df['refined_origin'] = met_df['refined_origin'].str.replace('Host', 'Coral')
met_df = met_df[met_df['refined_origin'] == 'Coral']

y = coral_df['scleractinia']

# 2. Define the 4 Feature Sets
# Load Ubiquity Data
ubiquity_df = pd.read_csv('/work/hs325/coral/data/ubiquity_summary.csv')
top_coral_ubiquity = ubiquity_df.sort_values(by='coral_percent', ascending=False).head(20)['metabolite'].tolist()
top_unique_ubiquity = ubiquity_df.sort_values(by='coral_minus_noncoral', ascending=False).head(20)['metabolite'].tolist()
scleractinia_ubiq_df = pd.read_csv("/work/hs325/coral/data/ubiquity_corals.csv")
top_scleractinia_ubiquity = scleractinia_ubiq_df.sort_values(by='percent_in_corals', ascending=False).head(20)['metabolite'].tolist()


# Load Previous Importance Data
imp_df = pd.read_csv("/work/hs325/coral/out/coralmets_ft_importance.csv")

# Ensure we get the feature names correct (assuming 'Feature' is the column name based on index_label)
# If 'Feature' isn't in columns, we assume the first column holds the names
feat_col = 'Feature' if 'Feature' in imp_df.columns else imp_df.columns[0]

top_xgb_imp = imp_df.sort_values(by='XGBoost_Importance', ascending=False).head(20)[feat_col].tolist()
top_rf_imp = imp_df.sort_values(by='RandomForest_Importance', ascending=False).head(20)[feat_col].tolist()

# Dictionary to iterate over
feature_sets = {
    "Coral_Ubiquity": top_coral_ubiquity,
    "Unique_Ubiquity": top_unique_ubiquity,
    "Scleractinia_Ubiquity": top_scleractinia_ubiquity,
    "XGB_Best20": top_xgb_imp,
    "RF_Best20": top_rf_imp
}

# 3. Initialize hyperparameters - saved values from existing train using same dataset on 100-iterations bayesian optimization
xgb_params = {
    'n_estimators': 558, 'learning_rate': 0.3, 'max_depth': 15,
    'subsample': 0.5, 'colsample_bytree': 1.0, 'gamma': 0.0,
    'min_child_weight': 1, 'random_state': 88, 'eval_metric': 'logloss', 'n_jobs': -1
}

rf_params = {
    'n_estimators': 343, 'max_depth': 39, 'max_features': 'sqrt',
    'min_samples_leaf': 9, 'min_samples_split': 19, 'random_state': 88, 'n_jobs': -1
}

# List to store results from each loop
dfs_to_merge = []

print("Starting Loop through Feature Sets...")

for set_name, feature_list in feature_sets.items():
    print(f"\n--- Processing Set: {set_name} ---")
    
    # Filter features for this specific set
    # We use intersection to ensure we only grab features that actually exist in coral_df columns
    valid_features = coral_df.columns.intersection(feature_list)
    X_subset = coral_df[valid_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=98, stratify=y
    )
    
    xgb_clf = XGBClassifier(**xgb_params)
    xgb_clf.fit(X_train, y_train)
    
    rf_clf = RandomForestClassifier(**rf_params)
    rf_clf.fit(X_train, y_train)
    
    xgb_clf = XGBClassifier(**xgb_params)
    xgb_clf.fit(X_train, y_train)
    
    rf_clf = RandomForestClassifier(**rf_params)
    rf_clf.fit(X_train, y_train)


    print(f"\n>>> Evaluation for Feature Set: {set_name} <<<")
    
    xgb_preds = xgb_clf.predict(X_test)
    print(f"--- {set_name} : XGBoost Report ---")
    print(classification_report(y_test, xgb_preds, digits=4))

    rf_preds = rf_clf.predict(X_test)
    print(f"--- {set_name} : RandomForest Report ---")
    print(classification_report(y_test, rf_preds, digits=4))
    
    print(f"Calculating Permutation Importance for {set_name}...")
    
    rf_result = permutation_importance(
        rf_clf, X_test, y_test, n_repeats=50, random_state=98, n_jobs=-1
    )
    rf_series = pd.Series(rf_result.importances_mean, index=valid_features, name=f"{set_name}_RF_Imp")
    
    gb_result = permutation_importance(
        xgb_clf, X_test, y_test, n_repeats=50, random_state=98, n_jobs=-1
    )
    gb_series = pd.Series(gb_result.importances_mean, index=valid_features, name=f"{set_name}_GB_Imp")
    
    set_df = pd.DataFrame({
        f"{set_name}_RF_Imp": rf_series,
        f"{set_name}_GB_Imp": gb_series
    })
    
    dfs_to_merge.append(set_df)

# 4. Merge all results into one CSV
print("\nMerging results...")
if dfs_to_merge:
    final_df = functools.reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), 
        dfs_to_merge
    )
    
    output_path = '/work/hs325/coral/out/coralmets_permutation_importance.csv'
    final_df.to_csv(output_path, index=True, index_label='Feature')
    print(f"Feature importances saved to '{output_path}'")
    print(final_df.head())
else:
    print("No results to save.")