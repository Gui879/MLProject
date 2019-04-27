'''
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
import inspect
import copy

class Optimizer:

    def __init__(self,optimizer):
        if type(optimizer) != str:
            raise ValueError('Optimizer parameter must be a string')
        elif optimizer not in ['gridSearch','bayesianOptimizer']:
                raise ValueError('Optimizer must be either gridSearch or bayesianOptimizer')
        if optimizer == 'gridSearch':
            self.optimizer = self._gridSearch
        else:
            self.optimizer = self._bayesianOptimizer

    def get(self):
        return self.optimizer

    def _gridSearch(self,pipeline, param_grid, metric, cv):
        clf_gscv = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(metric))
        return clf_gscv

    def _bayesianOptimizer(self, model, param_grid, metric, cv, seed = None):

        model_name = model.__name__
        n_param_grid = {}
        for key,value in param_grid.items():
            key = key.replace(model_name + '__', '')
            n_param_grid[key] = (0,len(value))

        def fit(self,X,y):
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
            for train_index, test_index in skf.split(X,y):
                train = X.iloc[train_index]
                test = X.iloc[test_index]
                def ob_function(**(self.param_grid)):

                    if seed != None:
                        self.pipeline = Pipeline([model_name, model()])
                    else:
                        self.pipeline = Pipeline([model_name, model(random_state=self.seed, **self.param_grid)])
                    self.pipeline.fit(X,y)
                    return self.metric(self.pipeline.fit(X, y))


                #b_optimizer = BayesianOptimization(f=ob_function, pbounds=self.param_grid, random_state=1)
                #b_optimizer.maximize()

        b_optimizer = Bayes_holder(fit,model, param_grid, n_param_grid, metric, cv, seed)
        return b_optimizer

class Bayes_holder:

    def __init__(self, fit,model, param_grid, n_param_grid, metric, cv, seed):
        self.model = model
        self.param_grid = param_grid
        self.metric = metric
        self.cv = cv
        self.test = fit
        self.n_param_grid = n_param_grid

    def fit(self,X,y):
        self.test(self,X,y)

'''