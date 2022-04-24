import pandas as pd
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
        
    # label encode the features
    for col in features:
        lbl = preprocessing.LabelEncoder()
        
        lbl.fit(df[col])
        
        df.loc[:, col] = lbl.transform(df[col])
    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # get training data
    x_train = df_train[features].values
    
    # get validation data
    x_valid = df_valid[features].values
    
    # init random forrest model
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