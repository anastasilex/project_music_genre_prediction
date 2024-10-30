import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from pickle import dump, load

def open_data(path="https://www.dropbox.com/scl/fi/5zy935lqpaqr9lat76ung/music_genre_train.csv?rlkey=ccovu9ml8pfi9whk1ba26zdda&dl=1"):
  df = pd.read_csv(path)
  return df

def split_data(df: pd.DataFrame):
  y = df["music_genre"]
  X = df.drop('music_genre', axis=1)
  return X, y

def copy_delete(df: pd.DataFrame):
  df_duplicates = df.copy()
  df_duplicates.drop(columns=['instance_id'], inplace=True)
  duplicates_index = df_duplicates[df_duplicates.duplicated(keep=False)].sort_values(by='track_name').index
  df = df.drop(index=duplicates_index)
  
  return df
  

def preprocess_data(df: pd.DataFrame):
  X = copy_delete(df)

  X_cat_cols = X.columns[((X.dtypes == 'object').values)]
  X_num_cols = X.columns[((X.dtypes != 'object').values)]

  num_preproc = Pipeline(steps=[
      ('num_inputer', SimpleImputer(strategy='median')),
      ('num_quantile_transf', QuantileTransformer(output_distribution='normal')),
      ('num_scaler', MinMaxScaler())
  ])

  cat_preproc = Pipeline(steps=[
    ('cat_inputer', SimpleImputer(strategy='most_frequent'))
  ])

  data_preproc = ColumnTransformer(transformers=[
      ("num", num_preproc, X_num_cols),
      ("cat", cat_preproc, X_cat_cols)
      ], remainder = 'passthrough')

  X = pd.DataFrame(data_preproc.fit_transform(X), columns=data_preproc.get_feature_names_out())

  X['tname_len'] = X['cat__track_name'].apply(lambda x: len(x)/100)
  X = X.drop(columns=['cat__obtained_date', 'cat__track_name', 'num__instance_id'])

  cat_list = ['cat__key', 'cat__mode']
  num_list=['num__danceability', 'num__speechiness',
          'num__acousticness', 'num__duration_ms', 'num__energy',
          'num__instrumentalness', 'num__liveness', 'num__loudness', 'num__tempo',
          'num__valence']

  cat_ohe = ColumnTransformer(transformers=[
      ('ohe', OneHotEncoder(drop='first', sparse_output=False), cat_list)
  ], remainder='passthrough')

  X_bin = pd.DataFrame(cat_ohe.fit_transform(X), columns=cat_ohe.get_feature_names_out(X.columns))
  X = pd.concat([X_bin, X[num_list]], axis=1)
  X = X.dropna()

  return X

def fit_and_save_model(X, y, path=r"model_weights.mw"):
    params_catboost = {
    'learning_rate': 0.07316264582704338,
    'depth': 5,
    'colsample_bylevel': 0.7746818822994564,
    'min_data_in_leaf': 8,
    'boosting_type': 'Ordered',
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.15583694636262965
}
    model = CatBoostClassifier(**params_catboost)
    model.fit(X, y)

    prediction = model.predict(X)
    accuracy = f1_score(prediction, y, average='micro')
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)
        
def load_model_and_predict(df, path=r"model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]

    return prediction        
               
if __name__ == "__main__":
    df = open_data()
    X, y = preprocess_data(df)
    fit_and_save_model(X, y)        