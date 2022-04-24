import pandas as pd
from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    # read folded data
    df = pd.read_csv('../input/cat-in-the-dat-ii/cat_train_folds.csv')
    
    # all targets are features except id, target and kfold columns
    features = [
        x for x in df.columns if x not in ['id', 'target', 'kfold']
    ]
    
    # fill all NaN values with None
    # note that I am converting all columns to strings
    # it doesn't matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")
    
    # get training data using kfolds 
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data using kfolds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # One hot encoder
    ohe = preprocessing.OneHotEncoder()
    
    # fit ohe on training + valid features
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    
    ohe.fit(full_data[features])

    # transform train and valid feats
    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    
    # initialize truncated svd
    svd = decomposition.TruncatedSVD(n_components=120)
    
    # fit svd on full sparse training data
    full_sparse = sparse.vstack((x_train, x_valid))
    svd.fit(full_sparse)
    
    # transform sparse training data 
    x_train = svd.transform(x_train)
    
    # transform sparse validation data
    x_valid = svd.transform(x_valid)
    
    # initialize random forrest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # fit data on training data
    model.fit(x_train, df_train.target.values)
    
    # predict on validation data
    # we need the probablity vals as we are
    # calculating auc 
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    # run function for fold = 0
    # we can just replace this number and
    # run this for any fold
 
    for fold_ in range(5):
        run(fold_)