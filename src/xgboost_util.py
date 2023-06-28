from hyperopt import Trials, fmin, space_eval, tpe, STATUS_OK, hp
from hyperopt.pyll import scope
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from src.data_preparation import CousinCrossValidation


def hyperopt_objective(space: dict, X_train, y_train, groups_train) -> dict:
    custom_splitter = CousinCrossValidation.split(X=X_train, y=y_train, groups=groups_train)
    model = XGBRegressor()

    for k, v in space.items():
        space[k] = [v]

    search = GridSearchCV(estimator=model,
                          param_grid=space,
                          scoring="neg_mean_absolute_error",
                          cv=custom_splitter,
                          verbose=0,
                          return_train_score=False)

    print(space)

    search.fit(X=X_train, y=y_train, groups=groups_train)
    return {'loss': -1.0 * search.best_score_, 'status': STATUS_OK}


def invoke_hyperopt(space: dict, X_train, y_train, groups_train, num_tries=20):
    trials = Trials()
    opt_fn = lambda s: hyperopt_objective(s, X_train, y_train, groups_train)

    fmin_result = fmin(fn=opt_fn, space=space, algo=tpe.suggest, max_evals=num_tries, trials=trials)
    return space_eval(space, fmin_result)

def invoke_hyperopt_with_default_space(X_train, y_train, groups_train, num_tries=20):
    return invoke_hyperopt(get_optimizer_space(), X_train, y_train, groups_train, num_tries)

def get_optimizer_space() -> dict:
    return hp.choice('classifier_type', [
    {
        'booster': 'gbtree',
        'max_depth': scope.int(hp.quniform('max_depth', 3, 18, 1)),
        'gamma': hp.uniform('gamma', 1, 9),
        'eta': hp.uniform('eta', 0.2, 0.5),
        'reg_alpha': hp.uniform('reg_alpha', 0, 4),
        'reg_lambda': hp.uniform('reg_lambda', 0, 4),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 0, 10, 1)),
    },
    #{ overfitting hell
    #    'booster': 'gblinear',
    #    'reg_lambda': hp.uniform('lin_reg_lambda', 0, 4),
    #    'reg_alpha': hp.uniform('lin_reg_alpha', 0, 4),
    #},
    {
        'booster': 'dart',
        'max_depth': scope.int(hp.quniform('max_depth_', 3, 18, 1)),
        'gamma': hp.uniform('gamma_', 1, 9),
        'eta': hp.uniform('eta_', 0.2, 0.5),
        'sample_type': hp.choice('sample_type', ['uniform', 'weighted']),
        'rate_drop': hp.uniform('rate_drop', 0, 1),
        'one_drop': hp.choice('one_drop', [0, 1]),
        'skip_drop': hp.uniform('skip_drop', 0, 1)
    }
])

def get_ideal_params() -> dict:
    return {
        'booster': 'dart',
        'eta': 0.29436931168604835,
        'gamma': 6.371289473173438,
        'max_depth': 6,
        'one_drop': 0,
        'rate_drop': 0.8163974793036589,
        'sample_type': 'uniform',
        'skip_drop': 0.036468243999770014
    }
