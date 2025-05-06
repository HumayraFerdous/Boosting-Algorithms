import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
data = pd.read_csv('Breast_Cancer_data.csv')
import warnings
warnings.filterwarnings("ignore")
#print(data.head())
#print(data.dtypes)
data = data.drop(['id','Unnamed: 32'],axis=1)
data['diagnosis']=data['diagnosis'].map({'M':0,'B':1}).astype(int)
#print(data.isnull().sum())
X = data.drop('diagnosis',axis=1)
y = data['diagnosis']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=0)

clf = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=-1,
               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
               n_estimators=50, n_jobs=-1, num_leaves=19, objective=None,
               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion metrics: \n',cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
c_r=classification_report(y_test, y_pred)
print('LightGBM classification report: \n',c_r)

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()



