import pandas

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from model import preproc

def hyperparameter_tuning(
    classifier,
    param_grid,
    X_train,
    y_train,
):

    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        n_jobs=6,
        cv=5,
    )

    grid_search.fit(X_train, y_train)
    print("Best score: " + str(grid_search.best_score_))
    print("Best params: " + str(grid_search.best_params_))

    return grid_search


def hpt_gradient_boosting(X_train, y_train):
    """
    https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
    """
    param_grid = {
        'learning_rate':[0.15,0.1,0.05,0.01],
        'n_estimators':[100,250,500,750,1000],
        'max_depth': list(range(3, 7)),
    }

    classifier = GradientBoostingClassifier()
    return hyperparameter_tuning(classifier, param_grid, X_train, y_train)


def hpt_xgboost(X_train, y_train):
    """  """
    param_grid = {
        'learning_rate':[0.1,0.75,0.05,0.025,0.01],
        'n_estimators':[100,250,500,750,1000],
        'max_depth': list(range(3, 8)),
    }

    classifier = XGBClassifier()
    return hyperparameter_tuning(classifier, param_grid, X_train, y_train)


if __name__ == '__main__':
    dataset = pandas.read_csv('data/train.csv')

    train_set, targets = preproc(dataset)

    # tuning_result = hpt_gradient_boosting()
    tuning_result = hpt_xgboost(train_set, targets)
