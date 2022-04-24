import pandas as pd
from sklearn import model_selection
import argparse
import model_dispatcher

def k_fold_cv(dataset_path, output_path, cv):
    df = pd.read_csv(dataset_path)
    
    df['kfold'] = -1 # add kfold column and fill vals with -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    kf = model_dispatcher.cv[cv]
    
    # Fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] =  fold
    
    df.to_csv(output_path, index=False)

def stratified_k_fold_cv(dataset_path, output_path, cv):
    df = pd.read_csv(dataset_path)
    
    df['kfold'] = -1 # add kfold column and fill vals with -1
    
    df = df.sample(frac=1).reset_index(drop=True)
    
    y = df.target.values
    
    kf = model_dispatcher.cv[cv]
    
    # Fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
        df.loc[val_, 'kfold'] =  fold
    
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--input_path',
        type=str
    )
    
    parser.add_argument(
        '--output_path',
        type=str
    )
    
    parser.add_argument(
        '--cv',
        type=str
    )
    
    args = parser.parse_args()
    if args.cv == 'kfold':
        k_fold_cv(args.input_path, args.output_path, args.cv)
    else:
        stratified_k_fold_cv(args.input_path, args.output_path, args.cv)