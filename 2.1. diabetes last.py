"""
Author  : Hüseyin Oğuzhan ŞAN
LinkedIn: https://www.linkedin.com/in/oguzhansan/
"""

################################################################
# Diabetes Cases
################################################################

# Business Problem

# In this project, we will develop a machine learning model that can
# predict whether Pima Indian Women in the dataset have diabetes or
# not. In addition, we will use to this model for Pipeline.

# Content of Variables:
# Pregnancies              - Number of pregnancies
# Glucose                  - Glucose concentration in the blood
# BloodPressure            - Blood Pressure
# SkinThickness            - Thickness of Skin
# Insulin                  - Level of insulin
# DiabetesPedigreeFunction - Diabetes parameters based on family history
# BMI                      - Body Mass Index
# Age                      - Age
# Outcome                  - Diabetic or non-diabetic ( 1 or 0 )

# 1. Exploratory Data Analysis (EDA)
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier

df = pd.read_csv("Diabetes/diabetes.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 40)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


################################################
# 1. Exploratory Data Analysis
################################################


def check_df(dataframe, head=5):
    print(10 * "#-#-" + "Shape".center(20) + 10 * "#-#-")
    print(dataframe.shape)
    print(10 * "#-#-" + "Info".center(20) + 10 * "#-#-")
    print(dataframe.info())
    print(10 * "#-#-" + "Head".center(20) + 10 * "#-#-")
    print(dataframe.head(head))
    print(10 * "#-#-" + "Tail".center(20) + 10 * "#-#-")
    print(dataframe.tail(head))
    print(10 * "#-#-" + "NA Values".center(20) + 10 * "#-#-")
    print(dataframe.isnull().sum())
    print(10 * "#-#-" + "Zero Values".center(20) + 10 * "#-#-")
    print((dataframe == 0).sum())
    print(10 * "#-#-" + "Nunique Values".center(20) + 10 * "#-#-")
    print(dataframe.nunique())
    print(10 * "#-#-" + "Describe".center(20) + 10 * "#-#-")
    print(dataframe.describe([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print(30 * "#-#-")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(3 * "#-#-" + numerical_col.center(20) + 3 * "#-#-")
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def target_density_est_with_num(dataframe, target, numerical_col):
    plt.figure(figsize=(15, 8))
    ax = sns.kdeplot(df[col][df.Outcome == 1], color="green", fill=True)
    sns.kdeplot(df[col][df.Outcome == 0], color="red", fill=True)
    plt.legend(['Has Diabete', 'Has no Diabete'])
    plt.xlim(-10, 150)
    plt.title("Diabetic Density of Numerical Variables")
    plt.show(block=True)


check_df(df)

# Split of variable types
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

# Examining categorical variables
for col in cat_cols:
    cat_summary(df, col)

# Examining numerical variables
for col in num_cols:
    num_summary(df, col, plot=True)

# Examining numerical variables with target
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

for col in num_cols:
    target_density_est_with_num(df, "Outcome", col)

# Correlation of all variables with each other
correlation_matrix(df, num_cols)

# Summary:
# There are only numerical variables in this dataset, and there are 768 observations, 9 variable available(Include 1 dependent variable.).
# It seems likely, data set have a normal distribution(Except insulin.) In addition, there are some hidden missing values in the data set.
# Finally, there is a categorical columns(Target).


################################################
# 2. Data Preprocessing & Feature Engineering
################################################


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

# Glucose                       5
# BloodPressure                35
# SkinThickness               227
# Insulin                     374
# BMI                          11

dimension_variable = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
# Actually, dimention variables should not be zero value. Impossible. We will fix that.

df1 = df.copy()
df1[dimension_variable] = df1[dimension_variable].replace(0,np.NaN)
check_df(df1)

# It works.

# Outlier Values (Before the Exchange Process)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df[dimension_variable])
plt.title("Before")

df[dimension_variable] = df[dimension_variable].replace(0, np.NaN)

check_df(df)

# Outlier Values (After the Exchange Process)
plt.subplot(1, 2, 2)
sns.boxplot(data=df1[dimension_variable])
plt.title("After")

check_df(df)

# NA Values
# Glucose                       5
# BloodPressure                35
# SkinThickness               227
# Insulin                     374
# BMI                          11

# The columns have low missing frequences and it observed a largely normal distribution.
df["Glucose"] = df["Glucose"].fillna(df["Glucose"].median())
df["BloodPressure"] = df["BloodPressure"].fillna(df["BloodPressure"].median())
df["BMI"] = df["BMI"].fillna(df["BMI"].median())

# The columns have high missing frequences and it observed abnormal distribution. We could KNN method for this problem.
dff = pd.get_dummies(df[["Insulin", "SkinThickness"]], drop_first=True)

dff.head()

# # Standardization of variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# # Implement the KNN method
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# # Undo the standardization of these variables
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["Insulin"] = dff["Insulin"]
df["SkinThickness"] = dff["SkinThickness"]

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Outlier Values Table
for col in df:
    plt.figure()
    sns.boxplot(x=df[col])

# experiment
df_new = df.copy()

for col in num_cols:
    print(col, check_outlier(df_new, col))

for col in num_cols:
    print(col, outlier_thresholds(df_new, col))

for col in num_cols:
    replace_with_thresholds(df_new, col)

# Actually, I want to shave a little bit.

for col in num_cols:
    print(col, outlier_thresholds(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Create the New Glucose Category (According to WHO)
df.loc[(df['Glucose'] < 70), 'GLUCOSE_CAT'] = "hypoglycemia"
df.loc[(df['Glucose'] >= 70) & (df['Glucose'] < 100), 'GLUCOSE_CAT'] = "normal"
df.loc[(df['Glucose'] >= 100) & (df['Glucose'] < 126), 'GLUCOSE_CAT'] = "over glucose"
df.loc[(df['Glucose'] >= 126), 'GLUCOSE_CAT'] = "hyperglycemia"

df.groupby("GLUCOSE_CAT").agg({"Outcome": ["mean", "count"]})

#                    Outcome
#                  mean count
# GLUCOSE_CAT
# hyperglycemia   0.593   297
# hypoglycemia    0.000    11
# normal          0.077   181
# over glucose    0.280   279

# Create the New Age Category (According to WHO)
df.loc[(df['Age'] >= 18) & (df['Age'] < 30), 'AGE_CAT'] = "young"
df.loc[(df['Age'] >= 30) & (df['Age'] < 45), 'AGE_CAT'] = "mature"
df.loc[(df['Age'] >= 45) & (df['Age'] < 65), 'AGE_CAT'] = "middle_age"
df.loc[(df['Age'] >= 65) & (df['Age'] < 95), 'AGE_CAT'] = "old_age"

df.groupby("AGE_CAT").agg({"Outcome": ["mean", "count"]})
#              Outcome
#                 mean count
# AGE_CAT
# mature_adult   0.494   239
# middle_age     0.530   117
# old_age        0.250    16
# young_adult    0.212   396

# Create a New Body Mass Index(BMI) Category (According to WHO)
df.loc[(df['BMI'] >= 16) & (df['BMI'] < 18.5), 'BMI_CAT'] = "weak"
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 25), 'BMI_CAT'] = "normal"
df.loc[(df['BMI'] >= 25) & (df['BMI'] < 30), 'BMI_CAT'] = "overweight"
df.loc[(df['BMI'] >= 30) & (df['BMI'] < 70), 'BMI_CAT'] = "obese"

df.groupby("BMI_CAT").agg({"Outcome": ["mean", "count"]})
#                Outcome
#               mean count
# BMI_CAT
# normal       0.069   102
# obese        0.458   483
# overweight   0.223   179
# weak         0.000     4

df.head()

# # Create a New BloodPressure Category (According to WHO)
df.loc[(df['BloodPressure'] < 70), 'BloodPressure_CAT'] = "low"
df.loc[(df['BloodPressure'] >= 70) & (df['BloodPressure'] < 90), 'BloodPressure_CAT'] = "normal"
df.loc[(df['BloodPressure'] >= 90), 'BloodPressure_CAT'] = "high"

df.groupby("BloodPressure_CAT").agg({"Outcome": ["mean", "count"]})

# # Create a Insulin Categorical variable (According to WHO)
df.loc[(df['Insulin'] < 120), 'INSULIN_CAT'] = "normal"
df.loc[(df['Insulin'] >= 120), 'INSULIN_CAT'] = "abnormal"

df.groupby("INSULIN_CAT").agg({"Outcome": ["mean", "count"]})

# # Create a Pregnancies Categorical variable
df.loc[(df['Pregnancies'] == 0), 'PREG_CAT'] = "unpregnant"
df.loc[(df['Pregnancies'] > 0) & (df['Pregnancies'] <= 5), 'PREG_CAT'] = "normal"
df.loc[(df['Pregnancies'] > 5) & (df['Pregnancies'] <= 10), 'PREG_CAT'] = "high"
df.loc[(df['Pregnancies'] > 10), 'PREG_CAT'] = "very high"

df.groupby("PREG_CAT").agg({"Outcome": ["mean", "count"]})


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [col for col in cat_cols if "Outcome" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)

#     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin    BMI  DiabetesPedigreeFunction    Age  Outcome  GLUCOSE_CAT_hypoglycemia  GLUCOSE_CAT_normal  GLUCOSE_CAT_over glucose  AGE_CAT_middle_age  AGE_CAT_old_age  AGE_CAT_young  BMI_CAT_obese  BMI_CAT_overweight  BMI_CAT_weak  BloodPressure_CAT_low  BloodPressure_CAT_normal  INSULIN_CAT_normal  PREG_CAT_normal  PREG_CAT_unpregnant  PREG_CAT_very high
# 0              6  148.000         72.000         35.000  159.200 33.600                     0.627 50.000        1                         0                   0                         0                   1                0              0              1                   0             0                      0                         1                   0                0                    0                   0
# 1              1   85.000         66.000         29.000  188.800 26.600                     0.351 31.000        0                         0                   1                         0                   0                0              0              0                   1             0                      1                         0                   0                1                    0                   0
# 2              8  183.000         64.000         29.153  155.548 23.300                     0.672 32.000        1                         0                   0                         0                   0                0              0              0                   0             0                      1                         0                   0                0                    0                   0
# 3              1   89.000         66.000         23.000   94.000 28.100                     0.167 21.000        0                         0                   1                         0                   0                0              1              0                   1             0                      1                         0                   1                1                    0                   0
# 4              0  137.000         40.000         35.000  168.000 43.100                     2.288 33.000        1                         0                   0                         0                   0                0              0              1                   0             0                      1                         0                   0                0                    1                   0
# ..           ...      ...            ...            ...      ...    ...                       ...    ...      ...                       ...                 ...                       ...                 ...              ...            ...            ...                 ...           ...                    ...                       ...                 ...              ...                  ...                 ...
# 763           10  101.000         76.000         48.000  180.000 32.900                     0.171 63.000        0                         0                   0                         1                   1                0              0              1                   0             0                      0                         1                   0                0                    0                   0
# 764            2  122.000         70.000         27.000  223.400 36.800                     0.340 27.000        0                         0                   0                         1                   0                0              1              1                   0             0                      0                         1                   0                1                    0                   0
# 765            5  121.000         72.000         23.000  112.000 26.200                     0.245 30.000        0                         0                   0                         1                   0                0              0              0                   1             0                      0                         1                   1                1                    0                   0
# 766            1  126.000         60.000         29.153  155.548 30.100                     0.349 47.000        1                         0                   0                         0                   1                0              0              1                   0             0                      1                         0                   0                1                    0                   0
# 767            1   93.000         70.000         31.000  127.800 30.400                     0.315 23.000        0                         0                   1                         0                   0                0              1              1                   0             0                      0                         1                   0                1                    0                   0
# [768 rows x 24 columns]

df.columns = [col.upper() for col in df.columns]

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if "OUTCOME" not in col]


######################################################
# 3. Base Models
######################################################

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier())]
    # ('CatBoost', CatBoostClassifier(verbose=False))

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

base_models(X, y, scoring="accuracy")

# accuracy: 0.7709 (LR) *
# accuracy: 0.7513 (KNN)
# accuracy: 0.7514 (SVC)
# accuracy: 0.6993 (CART)
# accuracy: 0.7644 (RF)
# accuracy: 0.7527 (Adaboost)
# accuracy: 0.7604 (GBM)
# accuracy: 0.7253 (XGBoost)
# accuracy: 0.7474 (LightGBM) *


######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]



def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

######################################################
# 5. Stacking & Ensemble Learning
######################################################

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)


######################################################
# 6. Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)



