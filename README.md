# Pipeline
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
