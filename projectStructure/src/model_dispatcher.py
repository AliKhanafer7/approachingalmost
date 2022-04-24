from sklearn import tree, ensemble
from sklearn import model_selection

models = {
    'decision_tree_gini': tree.DecisionTreeClassifier(criterion='gini'),
    'decision_tree_entropy': tree.DecisionTreeClassifier(criterion='entropy'),
    'rf': ensemble.RandomForestClassifier(),
}

cv = {
    'KFold': model_selection.KFold(n_splits=5),
    'stratifiedKFold': model_selection.StratifiedKFold(n_splits=5)
}