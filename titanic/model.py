"""
Some personal notes
========================


'Sex' seems to be the best predictor for survival on the titanic.

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

Input:
    'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001],
    'n_estimators':[100,250,500,750,1000,1250,1500,1750],
    'max_depth': list(range(3, 10)),

Best params: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 500}


"""


import math
import matplotlib.pyplot as plt
import numpy
import pandas
import pydotplus

from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.base import clone


columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
COLUMNS_TO_DROP = ['Name', 'Ticket', 'Cabin', 'Embarked']
CROSS_VALIDATION_K_FOLDS = 5


def write_solution(trained_model):
    dataset = pandas.read_csv('data/test.csv')
    test_set = preproc(dataset)

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

    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)

    # Sex to {0,1}
    label_binarizer = preprocessing.LabelBinarizer().fit(dataset['Sex'])
    dataset['Sex'] = label_binarizer.transform(dataset['Sex'])

    return dataset


def _scores_mean_accuracy(scores, clf_name):
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
    Manual Cross Validation
    """
    kfold = KFold(n_splits=5)

    classifier = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
    )

    cv_scores = []
    for train_fold_ixs, test_fold_ixs in kfold.split(train_set):

        trained_model = clone(classifier).fit(
            train_set.iloc[train_fold_ixs],
            targets.iloc[train_fold_ixs],
            # early_stopping_rounds=5,
            # evaluate `stopping` over the left out test Fold
            # eval_set=[train_set.iloc[test_fold_ixs], targets.iloc[test_fold_ixs]],
        ) 
        predictions = trained_model.predict(train_set.iloc[test_fold_ixs])
        cv_scores.append(
            trained_model.score(
                train_set.iloc[test_fold_ixs],
                targets.iloc[test_fold_ixs],
            )
        )

    cv_scores = numpy.array(cv_scores)
    print(_scores_mean_accuracy(cv_scores, 'XGBoost'))

    # clone again for readability
    final_model = clone(classifier).fit(train_set, targets)

    return final_model


if __name__ == '__main__':
    dataset = pandas.read_csv('data/train.csv')
    targets = dataset['Survived']

    train_set = preproc(dataset.drop('Survived', axis=1))

    trained_gradien_boosting_model = train_model(train_set, targets)
    trained_xgboost_model = XGBoost_training(train_set, targets)

    write_solution(trained_xgboost_model)
