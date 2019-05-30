"""
Some personal notes
========================


Gender seems to be the best predictor for survival on the titanic.

**Features correlation**
========================

Seems that First class correlates to:
- fare (obviously)
- sex (males tend to be in first class -> be richer?)
- Age (older people tend to be in better classes)

          Pclass       Sex       Age     SibSp     Parch      Fare
Pclass  1.000000  0.131900 -0.331339  0.083081  0.018443 -0.549500
Sex     0.131900  1.000000  0.084153 -0.114631 -0.245489 -0.182333
Age    -0.331339  0.084153  1.000000 -0.232625 -0.179191  0.091566
SibSp   0.083081 -0.114631 -0.232625  1.000000  0.414838  0.159651
Parch   0.018443 -0.245489 -0.179191  0.414838  1.000000  0.216225
Fare   -0.549500 -0.182333  0.091566  0.159651  0.216225  1.000000


**HPT**
========================

Gradient Boosting Input:
    'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
    'n_estimators':[100,250,500,750,1000,1250,1500,1750],
    'max_depth': list(range(3, 10)),

Best params: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 500}


XGBoost
-------

Feature importances: [
('Pclass', 0.1814381),
('Sex', 0.6286483),
('Age', 0.02732801),
('SibSp', 0.05935158),
('Parch', 0.012525526),
('Fare', 0.028575614),
('x0_C', 0.016267084),
('x0_Q', 0.013576938),
('x0_S', 0.03228884,)] 

"""


import math
import matplotlib.pyplot as plt
import numpy
import pandas
import pydotplus
import random
import yaml


import mlflow
import mlflow.sklearn
from contextlib import contextmanager
from sklearn import tree
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.base import clone

from my_plotting import plot_precision_recall
from my_plotting import plot_precision_recall_from_model


columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
COLUMNS_TO_DROP = ['Name', 'Ticket', 'Cabin'] #, 'Embarked']
CROSS_VALIDATION_K_FOLDS = 5


def write_solution(trained_model):
    dataset = pandas.read_csv('data/test.csv')
    test_set, _ = preproc(dataset)

    predictions = trained_model.predict(test_set)

    result = pandas.DataFrame(predictions, columns=['Survived'])
    result['PassengerId'] = dataset.index
    result.set_index('PassengerId', inplace=True)

    result.to_csv('data/my_answer.csv')


def preproc(dataset):
    """ Data preprocessing """

    # Not sure what to do with PassengerId and Names for now
    dataset.set_index('PassengerId', inplace=True)
    dataset.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

    # Imputation
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

    str_imputer = SimpleImputer(strategy='most_frequent')
    embarked = str_imputer.fit(dataset[['Embarked']]).transform(dataset[['Embarked']])
    dataset['Embarked'] = embarked

    ### CATEGORICAL DATA

    # Sex to {0,1}
    label_binarizer = preprocessing.LabelBinarizer().fit(dataset['Sex'])
    dataset['Sex'] = label_binarizer.transform(dataset['Sex'])

    # Embarked to One Hot Encoding
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(dataset[['Embarked']])

    one_hot_encoded = encoder.transform(dataset[['Embarked']]).toarray()
    dataset[encoder.get_feature_names()] = pandas.DataFrame(one_hot_encoded, index=dataset.index)
    dataset.drop('Embarked', axis=1, inplace=True)

    targets = None
    if 'Survived' in dataset:
        targets = dataset['Survived']
        dataset.drop('Survived', axis=1, inplace=True)

    return dataset, targets


def _scores_mean_accuracy(scores, clf_name):
    mlflow.log_metric('cv_mean_accuracy', scores.mean())
    return f"[{clf_name}] CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)


def cross_validation(classifier, train_set, targets):
    clf_name=classifier.__class__

    # Cross validation
    scores = cross_val_score(classifier, train_set, targets, cv=CROSS_VALIDATION_K_FOLDS)
    print(_scores_mean_accuracy(scores, clf_name))


def train_decision_tree(train_set, targets):
    classifier = tree.DecisionTreeClassifier()
    
    trained_model = cross_validation(classifier, train_set, targets)

    # Generate graphical representation of a decision tree
    dot_data = tree.export_graphviz(trained_model, out_file=None, feature_names=train_set.columns)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("data/decision_tree.pdf")
    mlflow.log_artifact('data/decision_tree.pdf')

    return trained_model


def train_model(train_set, targets):

    classifier = GradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        n_estimators=500,
    )

    cross_validation(classifier, train_set, targets)

    # AUROC score computation
    # scores = trained_model.predict_proba(train_set)[:, 1]
    # fpr, tpr, thresholds = roc_curve(targets, scores)
    # auroc_score = roc_auc_score(targets, scores)
    # print("AUROC score of the model: " + str(auroc_score))


    # cross_validation clones models already but clone again for readability
    final_model = clone(classifier).fit(train_set, targets)

    return final_model


def XGBoost_training(train_set, targets):
    """
    XGBoost training with 'manual' Cross Validation
    """
    N_ESTIMATORS = 280
    LEARNING_RATE = 0.025
    MAX_DEPTH=5

    mlflow.log_param('model_type', 'XGBoost')
    mlflow.log_param('CROSS_VALIDATION_K_FOLDS', CROSS_VALIDATION_K_FOLDS)
    mlflow.log_param('N_ESTIMATORS', N_ESTIMATORS)
    mlflow.log_param('LEARNING_RATE', LEARNING_RATE)
    mlflow.log_param('MAX_DEPTH', MAX_DEPTH)

    kfold = KFold(n_splits=CROSS_VALIDATION_K_FOLDS)

    classifier = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
    )

    cv_scores = []
    for (fold_n, (train_fold_ixs, test_fold_ixs)) in enumerate(kfold.split(train_set)):

        trained_model = clone(classifier).fit(
            train_set.iloc[train_fold_ixs],
            targets.iloc[train_fold_ixs],
            # early_stopping_rounds=5,
            # evaluate `stopping` over the left out test Fold
            # eval_set=[train_set.iloc[test_fold_ixs], targets.iloc[test_fold_ixs]],
        ) 
        cv_scores.append(
            trained_model.score(
                train_set.iloc[test_fold_ixs],
                targets.iloc[test_fold_ixs],
            )
        )

        plot_precision_recall_from_model(
            trained_model,
            train_set.iloc[test_fold_ixs],
            targets.iloc[test_fold_ixs],
            False,
            'data/precision_recall_cvfold_{}.png'.format(fold_n),
        )

    cv_scores = numpy.array(cv_scores)
    print(_scores_mean_accuracy(cv_scores, 'XGBoost'))

    # clone again for readability
    final_model = clone(classifier).fit(train_set, targets)

    plot_precision_recall_from_model(final_model, train_set, targets, 'data/precision_recall.png')
    mlflow.log_artifact('data/precision_recall.png')

    return final_model

@contextmanager
def setup_mlflow():
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('Titanic')

    random_hash = random.getrandbits(128)
    with mlflow.start_run(run_name=str(random_hash)):
        yield


if __name__ == '__main__':
    with setup_mlflow():
        dataset = pandas.read_csv('data/train.csv')
        # targets = dataset['Survived']

        train_set, targets = preproc(dataset)

        # trained_gradien_boosting_model = train_model(train_set, targets)
        trained_xgboost_model = XGBoost_training(train_set, targets)

        mlflow.sklearn.log_model(trained_xgboost_model, 'titanic_model')

        with open('feature_importances.yaml', 'w') as feat_imp_file:
            yaml.safe_dump(dict(zip(
                train_set.columns.to_list(),
                map(str, trained_xgboost_model.feature_importances_),
            )), feat_imp_file)

        mlflow.log_artifact('feature_importances.yaml')

        print("Feature importances: " + str(list(zip(
            train_set.columns.to_list(),
            trained_xgboost_model.feature_importances_
        ))))

        write_solution(trained_xgboost_model)
