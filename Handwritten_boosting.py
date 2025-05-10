import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


df = pd.read_csv("train.csv")
df['binary_target'] = (df['label'] == 0).astype(int)
X = df.drop(columns=['label', 'binary_target'])
y_binary = df['binary_target']
y_multiclass = df['label']

mi = mutual_info_classif(X, y_binary, random_state=42)
mi_scores = pd.Series(mi, index=X.columns)
mi_selected = mi_scores.sort_values(ascending=False).head(30).index.tolist()

X_mi = X[mi_selected]
corr_matrix = X_mi.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
final_features = [f for f in mi_selected if f not in to_drop]

print(f"Selected Features after Mutual Info & Correlation: {final_features}")

X_train, X_test, y_train, y_test = train_test_split(X[final_features], y_binary, test_size=0.2, random_state=42)

lgb = LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)
y_pred_bin = lgb.predict(X_test)

print("\n=== First Layer: Binary Classification (Digit 0 vs Others) ===")
print(classification_report(y_test, y_pred_bin, target_names=["Others", "Digit 0"]))

df['bin_prob'] = lgb.predict_proba(X[final_features])[:, 1]
threshold = 0.5
df_layer2 = df[df['bin_prob'] < threshold]

X_layer2 = df_layer2[final_features]
y_layer2 = df_layer2['label']

X_layer2 = X_layer2[y_layer2 != 0]
y_layer2 = y_layer2[y_layer2 != 0]
label_map = {old: new for new, old in enumerate(sorted(y_layer2.unique()))}
reverse_map = {v: k for k, v in label_map.items()}
y_layer2_mapped = y_layer2.map(label_map)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_layer2, y_layer2_mapped, test_size=0.2, random_state=42
)

xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=10,
    eval_metric='mlogloss',
    use_label_encoder=True,
    random_state=42
)
xgb.fit(X2_train, y2_train)
y_pred_multi = xgb.predict(X2_test)
y_pred_original = [reverse_map[pred] for pred in y_pred_multi]
y_true_original = [reverse_map[true] for true in y2_test]
print("\n=== Second Layer: Multiclass Classification (Digits 1–9) ===")
print(classification_report(y_true_original, y_pred_original))