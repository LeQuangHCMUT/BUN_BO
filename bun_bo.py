import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Đọc dữ liệu
train   = pd.read_csv(r"F:\Python_Dataset\Mechine Learning\Dataset\BUN_DO\data_train_demo_new.csv")
test    = pd.read_csv(r"F:\Python_Dataset\Mechine Learning\Dataset\BUN_DO\data_test_demo_new.csv")
train   = train.drop("action", axis = 1)
test    = test.drop("action", axis = 1)

target = "operation"

# Kiểm tra dữ liệu
print("Kích thước dữ liệu train:", train.shape)
print("Kích thước dữ liệu test:", test.shape)

# Chuyển cột target thành kiểu category
train[target] = train[target].astype('category')
test[target] = test[target].astype('category')

# Xử lý giá trị thiếu
train.dropna(inplace=True)
test.dropna(inplace=True)

# Chia thành feature và label
X_train = train.drop(target, axis=1)
y_train = train[target]

X_test = test.drop(target, axis=1)
y_test = test[target]

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Trước SMOTE:", pd.Series(y_train_encoded).value_counts())
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)
print("Sau SMOTE:", pd.Series(y_train_resampled).value_counts())

print("Nhãn trong y_train_resampled:", np.unique(y_train_resampled))
print("Nhãn trong y_test_encoded:", np.unique(y_test_encoded))

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    # "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
}

for model_name, model in models.items():
    print(f"\n🔍 Đang huấn luyện mô hình: {model_name}")
    model.fit(X_train_resampled, y_train_resampled)

    y_pred_encoded = model.predict(X_test)

    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_original = label_encoder.inverse_transform(y_test_encoded)

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
        if len(np.unique(y_test_encoded)) == 2:
            auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1], average='macro')
        else:
            auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr')
        print(f"AUROC của {model_name}: {auc:.6f}")
    else:
        print(f"{model_name} không hỗ trợ predict_proba().")

    print(classification_report(y_test_original, y_pred))

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test_original, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



# AUROC của Random Forest: 0.936174
#               precision    recall  f1-score   support
#
#          100       0.50      0.96      0.66     14980
#          200       0.80      0.47      0.59     22388
#          300       0.68      0.80      0.74     23765
#          400       0.69      0.81      0.74     12987
#          500       0.66      0.44      0.53     14822
#          600       0.75      0.10      0.18      6453
#          700       0.81      0.81      0.81     28081
#          800       0.53      0.74      0.62     10401
#          900       0.87      0.72      0.79      7552
#         1000       0.77      0.66      0.71     12004
#         8100       0.12      0.04      0.06      2445
#
#     accuracy                           0.68    155878
#    macro avg       0.65      0.60      0.58    155878
# weighted avg       0.70      0.68      0.66    155878


# AUROC của Logistic Regression: 0.771616
#               precision    recall  f1-score   support
#
#          100       0.38      0.49      0.42     14980
#          200       0.43      0.13      0.20     22388
#          300       0.53      0.12      0.19     23765
#          400       0.73      0.51      0.60     12987
#          500       0.23      0.17      0.20     14822
#          600       0.11      0.31      0.16      6453
#          700       0.60      0.55      0.57     28081
#          800       0.22      0.68      0.33     10401
#          900       0.46      0.20      0.28      7552
#         1000       0.30      0.53      0.38     12004
#         8100       0.01      0.01      0.01      2445
#
#     accuracy                           0.35    155878
#    macro avg       0.36      0.34      0.30    155878
# weighted avg       0.43      0.35      0.34    155878


# AUROC của KNN (K-Nearest Neighbors): 0.661183
#               precision    recall  f1-score   support
#
#          100       0.34      0.54      0.42     14980
#          200       0.27      0.26      0.26     22388
#          300       0.31      0.19      0.23     23765
#          400       0.50      0.34      0.41     12987
#          500       0.17      0.15      0.16     14822
#          600       0.07      0.12      0.09      6453
#          700       0.59      0.48      0.53     28081
#          800       0.17      0.19      0.18     10401
#          900       0.35      0.36      0.36      7552
#         1000       0.32      0.36      0.34     12004
#         8100       0.02      0.05      0.03      2445
#
#     accuracy                           0.31    155878
#    macro avg       0.28      0.28      0.27    155878
# weighted avg       0.34      0.31      0.32    155878

# AUROC của XGBoost: 0.948898
#               precision    recall  f1-score   support
#
#          100       0.58      0.96      0.72     14980
#          200       0.97      0.42      0.58     22388
#          300       0.71      0.54      0.61     23765
#          400       0.56      0.90      0.69     12987
#          500       0.56      0.51      0.54     14822
#          600       0.94      0.15      0.26      6453
#          700       0.79      0.82      0.80     28081
#          800       0.60      0.85      0.70     10401
#          900       0.83      0.85      0.84      7552
#         1000       0.70      0.85      0.77     12004
#         8100       0.13      0.08      0.10      2445
#
#     accuracy                           0.68    155878
#    macro avg       0.67      0.63      0.60    155878
# weighted avg       0.72      0.68      0.66    155878