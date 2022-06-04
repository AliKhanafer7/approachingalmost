import copy
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb

def mean_target_encoding(data):
    
    #make a copy of the data
    df = copy.deepcopy(data)
    
    # list of numerical columns
    num_cols = [
        "fnlwgt",
        "age",
        "capital.gain",
        "capital.loss",
        "hours.per.week"
    ]
    
    # map targets to 0s and 1s
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    
    df.loc[:, 'target'] = df.target.map(target_mapping)
    
    # all columns are features except for kfold and target
    features = [
        f for f in df.columns if f not in ("kfold", "target")
        and f not in num_cols
    ]
    
    # fill all NaN values with NONE
    # note that I am converting all columns to strings
    # it doesn't matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
        
    # label encode
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:,col] = lbl.transform(df[col])
    
    # a list to store 5 validation dataframes
    encoded_dfs = []
    
    # go over all folds
    for fold in range(5):
        # featch training and validation data
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        # for all feature columns
        for column in features:
            # create dict of category:mean target
            mapping_dict = dict(
                df_train.groupby(column)["target"].mean()
            )
            
            # column_enc is the new column we have with mean encoding
            df_valid.loc[
                :, column + "_enc"
            ] = df_valid[column].map(mapping_dict)

        # append to our list of encoded validation dfs
        encoded_dfs.append(df_valid)
        
        # create full df again and return
        encoded_df = pd.concat(encoded_dfs, axis=0)

    return encoded_df

def run(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    features = [
        f for f in df.columns if f not in ('kfold', 'target')
    ]
    
    x_train = df_train[features].values
    x_valid = df_valid[features].values
    
    model = xgb.XGBClassifier(
        n_jobs=-1,
        max_depth=7
    )
    
    model.fit(x_train, df_train.target.values)
    
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    print(f'Fold = {fold}, AUC={auc}')
    
if __name__ == '__main__':
    df = pd.read_csv('../input/adult_folds.csv')

    # create mean target encoded categories and
    # munge data
    df = mean_target_encoding(df)

    for fold_ in range(5):
        run(df, fold_)