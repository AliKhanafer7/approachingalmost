import config
import model_dispatcher
import os
import argparse
import joblib
import pandas as pd
from sklearn import tree, metrics

def run(fold, model):
    
    # read csv
    df = pd.read_csv(config.TRAINING_FILE)
    
    # train data is data that isn't part of fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # validation data is data that is part of fold
    df_validation = df[df.kfold == fold].reset_index(drop=True)
    
    # Drop target from training data and assign it to new variable
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values
    
    # Drop target from validation data and assign it to new variable
    x_validation = df_validation.drop('label', axis=1).values
    y_validation = df_validation.label.values
    
    # Initialize simple tree classifier
    clf = model_dispatcher.models[model]
    
    # Fit on training data
    clf.fit(x_train, y_train)
    
    # Make predictions on validation data
    predictions = clf.predict(x_validation)
    
    # Calculate and print accuracy
    acc = metrics.accuracy_score(y_validation, predictions)
    
    print(f"Accuracy of fold {fold}: {acc}")
    
    # Save model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f'{model}_{fold}.bin')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--fold',
        type=int
    )
    
    parser.add_argument(
        '--model',
        type=str
    )
    
    args = parser.parse_args()
    
    run(fold=args.fold, model=args.model)