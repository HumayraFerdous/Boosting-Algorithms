import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
data = pd.read_csv('winequality-red.csv')
#print(data.head())
print(data.dtypes)
#print(data.isnull().sum())
print(data['quality'].value_counts())
data['quality_binary'] = data['quality'].apply(lambda x:1 if x>=7 else 0)
X = data.drop(['quality', 'quality_binary'], axis=1)
y = data['quality_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier( n_estimators=130,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=17,
    min_child_samples=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42, force_col_wise=True)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LightGBM - Confusion Matrix")
plt.show()