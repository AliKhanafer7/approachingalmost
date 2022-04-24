from sklearn import datasets, manifold, tree, metrics, model_selection, preprocessing, linear_model
import pandas as pd
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
    
    # initialize logistic reg
    model = linear_model.LogisticRegression()
    
    # fit on training data
    model.fit(x_train, df_train.target.values)
    
    # predict on validation data. This will output probabilities, 
    # since our evaluation metric is the AUC
    valid_preds = model.predict_proba(x_valid)[:,1]
    
    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    print(auc)

if __name__ == "__main__":
    # run function for fold = 0
    # we can just replace this number and
    # run this for any fold
 
    for fold_ in range(5):
        run(fold_)