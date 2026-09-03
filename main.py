import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(nums_attribs,cat_attribs):
    num_pipeline = Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("impute",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,nums_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    data = pd.read_csv("job_salary_prediction_dataset.csv")
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(data,data["experience_years"]):
        df = data.loc[test_index].to_csv("input.csv",index=False)
        sal = data.loc[train_index]
    sal_labels = sal["salary"].copy()
    sal_features = sal.drop("salary",axis=1)
    sal_num = sal_features.select_dtypes(include=[np.number]).columns.tolist()
    sal_cat = sal_features.select_dtypes(exclude=[np.number]).columns.tolist()
    pipeline = build_pipeline(sal_num,sal_cat)
    sal_prepared = pipeline.fit_transform(sal_features)
    model = RandomForestRegressor(random_state=42)
    model.fit(sal_prepared,sal_labels)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained and saved")
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["salary"] = predictions
    input_data["salary"].to_csv("output.csv",index=False)
    print("Inference complete and output saved to output.csv")