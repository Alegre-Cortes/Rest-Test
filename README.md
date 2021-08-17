# Rest-Test
Light interface with a collection of basic tools to explore datasets

**Dependencies**

PyQt5,
numpy,
matplotlib,
scipy,
umap-learn,
sklearn

## Summary 

Rest&Test is a light interface to explore the parametrization of a dataset. You can either load your owns parameters + labels, or a matrix of time series. In the last case, you need to provide a feature extraction pipeline to fill the function *core_functions.compute_parameters*.

By default, the used classifier is a Gaussian Proccess with a radial-basis function kernel. If another classifier is prefered, it can be easily substituted in *core_functions*.

## Required data format

Data should be presented as a python dictionary with the following keys:

['Labels'], which includes the labels that will be used for the suppervised classification.

['Data'] **OR** ['Parameters'], depending whether we want to load a dataset or a set of parameters.

Note that it is necessary to add the ['Labels'] key, otherwise the GUI will not work.

## GUI explanation

![Interface](https://github.com/Alegre-Cortes/Rest-Test/blob/main/Interface.PNG)

* _**Load data**_: Loads the dataset/parameters + labels dictionary.
* _**Features to plot**_: Space to indicate the number of the parameters to plot and study its distribution (2 parameters only).
* _**Dimensions to use for classification**_: Space to indicate the number of the parameters that will be used to classify using GP.
* _**Trace(s) to plot**_: Space to indicate the number of the traces that will be ploted, if a matrix of time series was loaded.
* _**PCA**: Performs PCA_: using the loaded parameters, returns the explained variance, a visualization of the 3 first PC and a tag with the accuray in the classification.
* _**tSNE**_: Performs tSNE using the loaded paramters, returns the 2D tSNE space, the trained GP and the accuracy of classification.
* _**Classify**_: Performs supervised classification using the features selected, shows the confusion matrix.
* _**Recursive Feature Extraction**_: Performs Recursive Feature Elimination analysis, returns the order of features in the workspace and plots the 3 first parameters with a tag indicating the accuracy.
* _**Plot trace**_: Plots the indicated time series.
