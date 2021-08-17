# Rest-Test
Light interface with a collection of basic tools to explore datasets

**Dependencies**

PyQt5,
numpy,
matplotlib,
scipy,
umap-learn,
sklearn

# Summary 

Rest&Test is a light interface to explore the parametrization of a dataset. You can either load your owns parameters + labels, or a matrix of time series, in which case the parameters extracted in [this paper](https://elifesciences.org/articles/60580) will be used. They are directed to characterize intracellular recordings of Slow Wave Oscillation, so they may not be of use in many situations. The function to extract the parameters *core_functions.compute_parameters* can be easily substituted to add your own feature extraction to the pipeline.

By default, the used classifier is a Gaussian Proccess with a radial-basis function ernel. If another classifier is prefered, it can be easily substituted in *core_functions*.

# Required data format

Data should be presented as a python dictionary with the following keys:

['Labels'], which includes the labels that will be used for the suppervised classification

['Data'] **OR** ['Parameters'], depending whether we want to load a dataset or a set of parameters

# GUI explanation

![Interface](https://github.com/Alegre-Cortes/Rest-Test/blob/main/Interface.PNG)

* **Load data**: Loads the dataset/parameters + labels dictionary.
* **Features to plot**: Space to indicate the number of the parameters to plot and study its distribution (2 parameters only).
* **Dimensions to use for classification**: Space to indicate the number of the parameters that will be used to classify using GP.
* **Trace(s) to plot**: Space to indicate the number of the traces that will be ploted, if a matrix of time series was loaded.
* **PCA**: Performs PCA using the loaded parameters, returns the explained variance, a visualization of the 3 first PC and a tag with the accuray in the classification.
* **tSNE**: Performs tSNE using the loaded paramters, returns the 2D tSNE space, the trained GP and the accuracy of classification.
* **Classify**: Performs supervised classification using the features selected, shows the confusion matrix.
* **Recursive Feature Extraction**: Performs Recursive Feature Elimination analysis, returns the order of features in the workspace and plots the 3 first parameters with a tag indicating the accuracy.
* **Plot trace**: Plots the indicated time series.
