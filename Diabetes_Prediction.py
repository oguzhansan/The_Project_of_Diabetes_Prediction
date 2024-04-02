"""
Author  : Hüseyin Oğuzhan ŞAN
LinkedIn: https://www.linkedin.com/in/oguzhansan/
"""


################################################
# Diabetes Machine Learning
################################################

import joblib
import pandas as pd

df = pd.read_csv("Diabetes/diabetes.csv")

random_user = df.sample(1, random_state=45)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)

from diabetes_pipeline_last import diabetes_data_prep

X, y = diabetes_data_prep(df)

random_user = X.sample(1, random_state=50)

new_model = joblib.load("voting_clf.pkl")

new_model.predict(random_user)
