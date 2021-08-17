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

# Required data format

Data should be presented as a python dictionary with the following keys:

['Labels'], which includes the labels that will be used for the suppervised classification

['Data'] **OR** ['Parameters'], depending whether we want to load a dataset or a set of parameters

# GUI explanation

![Interface](https://github.com/Alegre-Cortes/Rest-Test/blob/main/Interface.PNG)

* **Load data**: Loads the dataset/parameters + labels dictionary
* **Features to plot**:
* **Dimensions to use for classification**:
* **Trace(s) to plot**:
* **PCA**:
* **tSNE**:
* **Classify**:
* **Recursive Feature Extraction**:
* **Plot trace**:
