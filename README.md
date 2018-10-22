# Pipeline and Grid Search
Pipeline technique is useful tool for managing machine learning workflows.

A typical machine learning task generally involves data preparation to varying degrees. We won't get into the wide array of activities which make up data preparation here, but there are many. Such tasks are known for taking up a large proportion of time spent on any given machine learning task.

After a dataset is cleaned up from a potential initial state of massive disarray, however, there are still several less-intensive yet no less-important transformative data preprocessing steps such as feature extraction, feature scaling, and dimensionality reduction, to name just a few.

Maybe your preprocessing requires only one of these tansformations, such as some form of scaling. But maybe you need to string a number of transformations together, and ultimately finish off with an estimator of some sort. This is where Scikit-learn Pipelines can be helpful.

Scikit-learn's Pipeline class is designed as a manageable way to apply a series of data transformations followed by the application of an estimator

That's it. Ultimately, this simple tool is useful for:

-Convenience in creating a coherent and easy-to-understand workflow

-Enforcing workflow implementation and the desired order of step applications
Reproducibility

-Value in persistence of entire pipeline objects (goes to reproducibility and convenience)


Grid Search

Another simple yet powerful technique we can pair with pipelines to improve performance is grid search, which attempts to optimize model hyperparameter combinations. 

Exhaustive grid search -- as opposed to alternate hyperparameter combination optimization schemes such as randomized optimization -- tests and compares all possible combinations of desired hyperparameter values, an exercise in exponential growth.

The trade-off in what could end up being exorbitant run times would (hopefully) be the best optimized model possible.

The grid search provided by GridSearchCV exhaustively generates candidates from a grid of parameter values specified with the param_grid parameter. For instance, the following param_grid:

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
 

specifies that two grids should be explored: one with a linear kernel and C values in [1, 10, 100, 1000], and the second one with an RBF kernel, and the cross-product of C values ranging in [1, 10, 100, 1000] and gamma values in [0.001, 0.0001].

