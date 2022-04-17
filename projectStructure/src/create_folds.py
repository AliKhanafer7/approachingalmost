import pandas as pd
from sklearn import model_selection
def k_fold_cv(dataset_path, output_path):
    df = pd.read_csv(dataset_path)
    
    df['kfold'] = -1 # add kfold column and fill vals with -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    kf = model_selection.KFold(n_splits=5)
    
    # Fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] =  fold
    
    df.to_csv(output_path, index=False)

k_fold_cv('../input/mnist_train.csv', '../input/mnist_train_folds.csv')