# BANKRUPTCY PREDICTION - EDA + DECISION TREE + RANDOM FOREST
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score

# Dataset path (for Colab after extracting the ZIP)
LOCAL_CSV = 'bankruptcy_data/data.csv'

# 1. LOAD DATASET
if os.path.exists(LOCAL_CSV):
    df = pd.read_csv(LOCAL_CSV)
else:
    from ucimlrepo import fetch_ucirepo
    bankruptcy = fetch_ucirepo(id=572)
    df = pd.concat([bankruptcy.data.features, bankruptcy.data.targets], axis=1)
    if 'Bankrupt?' not in df.columns:
        df.rename(columns={df.columns[-1]: 'Bankrupt?'}, inplace=True)

print('Dataset loaded successfully!')
print('Shape:', df.shape)

# 2. EDA
print('\nFirst 5 rows:'); display(df.head())
print('\nShape:', df.shape)
print('\nData types:'); print(df.dtypes)
print('\nMissing values:', df.isnull().sum().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('\nStatistical summary:'); display(df.describe())

target = 'Bankrupt?'
print('\nTarget distribution:'); print(df[target].value_counts())
print('\nTarget percentages:'); print(df[target].value_counts(normalize=True) * 100)

plt.figure(figsize=(6,4)); sns.countplot(x=df[target]); plt.title('Bankruptcy Class Distribution'); plt.xlabel('Bankrupt? (0 = No, 1 = Yes)'); plt.ylabel('Number of Companies'); plt.tight_layout(); plt.show()

constant_features = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print('\nConstant features:', constant_features)

df_corr = df.replace([np.inf, -np.inf], np.nan)
correlations = df_corr.corr(numeric_only=True)[target].drop(target)
top_correlations = correlations.abs().sort_values(ascending=False).head(15)
print('\nTop 15 features related to bankruptcy:')
for feature in top_correlations.index:
    print(f'{feature}: {correlations[feature]:.4f}')

top_features = top_correlations.index.tolist()
plt.figure(figsize=(10,7)); correlations.loc[top_features].sort_values().plot(kind='barh'); plt.title('Top 15 Features by Absolute Correlation with Bankruptcy'); plt.xlabel('Correlation with Bankrupt?'); plt.tight_layout(); plt.show()

heatmap_data = df_corr[top_features + [target]].corr()
plt.figure(figsize=(10,8)); sns.heatmap(heatmap_data, annot=False, cmap='coolwarm', center=0); plt.title('Correlation Heatmap - Top Features'); plt.tight_layout(); plt.show()

for feature in top_features[:4]:
    plt.figure(figsize=(6,4)); sns.boxplot(x=df[target], y=df[feature]); plt.title(f'{feature} vs Bankruptcy'); plt.xlabel('Bankrupt? (0 = No, 1 = Yes)'); plt.ylabel(feature); plt.tight_layout(); plt.show()

# 3. PREPROCESSING
X = df.drop(columns=[target]).select_dtypes(include=[np.number]).copy()
y = df[target].astype(int).copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.loc[:, X.nunique(dropna=False) > 1]
X = X.fillna(X.median())

# 4. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print('\nTraining samples:', len(X_train)); print('Testing samples:', len(X_test))

# 5. DECISION TREE
dt_model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test); dt_prob = dt_model.predict_proba(X_test)[:,1]
dt_precision = precision_score(y_test, dt_pred, zero_division=0); dt_recall = recall_score(y_test, dt_pred, zero_division=0); dt_f1 = f1_score(y_test, dt_pred, zero_division=0); dt_ap = average_precision_score(y_test, dt_prob)
print('\nDECISION TREE'); print(classification_report(y_test, dt_pred, digits=4)); print('Confusion matrix:\n', confusion_matrix(y_test, dt_pred)); print('Average Precision:', dt_ap)

# 6. RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test); rf_prob = rf_model.predict_proba(X_test)[:,1]
rf_precision = precision_score(y_test, rf_pred, zero_division=0); rf_recall = recall_score(y_test, rf_pred, zero_division=0); rf_f1 = f1_score(y_test, rf_pred, zero_division=0); rf_ap = average_precision_score(y_test, rf_prob)
print('\nRANDOM FOREST'); print(classification_report(y_test, rf_pred, digits=4)); print('Confusion matrix:\n', confusion_matrix(y_test, rf_pred)); print('Average Precision:', rf_ap)

# 7. PRECISION-RECALL CURVE
p1, r1, _ = precision_recall_curve(y_test, dt_prob); p2, r2, _ = precision_recall_curve(y_test, rf_prob)
plt.figure(figsize=(8,6)); plt.plot(r1,p1,label=f'Decision Tree (AP={dt_ap:.4f})'); plt.plot(r2,p2,label=f'Random Forest (AP={rf_ap:.4f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# 8. FEATURE IMPORTANCE
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print('\nTop 15 Random Forest features:'); print(importance.head(15))
plt.figure(figsize=(10,7)); importance.head(15).sort_values().plot(kind='barh'); plt.title('Top 15 Random Forest Feature Importances'); plt.xlabel('Importance'); plt.tight_layout(); plt.show()

# 9. DECISION TREE VISUALIZATION
plt.figure(figsize=(22,10)); plot_tree(dt_model, feature_names=X.columns, class_names=['Non-Bankrupt','Bankrupt'], filled=True, max_depth=3, fontsize=7); plt.title('Decision Tree (First 3 Levels)'); plt.tight_layout(); plt.show()

# 10. MODEL COMPARISON
results = pd.DataFrame({'Model':['Decision Tree','Random Forest'],'Precision':[dt_precision,rf_precision],'Recall':[dt_recall,rf_recall],'F1-Score':[dt_f1,rf_f1],'Average Precision':[dt_ap,rf_ap]})
print('\nMODEL COMPARISON'); display(results)
print('\nProject execution completed.')
