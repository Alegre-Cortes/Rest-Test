# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:07:30 2019

@author: ramon.reig
"""
from scipy.signal import find_peaks 
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

def load_data():
# Function to load the labels + Time series/Parameters
    options = QtWidgets.QFileDialog.Options() 
    options |= QtWidgets.QFileDialog.DontUseNativeDialog 
    fileName,_ = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()","", options=options)
    fileName
    aux = np.load(fileName,allow_pickle=True)
    aux = aux.item()
    np.disp('Hello')
    layer = aux['Labels']
    if ('Data' in aux):
        dataset = aux['Data']
        parameters = compute_parameters(dataset, layer)               
    if ('Parameters' in aux):
        parameters = aux['Parameters']
        print('testing')
        dataset = []
    print(np.shape(dataset))
    return dataset, parameters, fileName, layer





def compute_parameters(dataset, layer):
# Empty function, ready to introduce a pipeline to perform feature extraction

      return parameters                    

       

def rfe(parameters, layer):
# Performs Recursive Feature Elimination, stopping with the 3 most useful, and using a SVC
# with a linear kernel    
      estimator = SVC(kernel="linear", C=1)
      selector = RFE(estimator, 3, step=1)
      [a,b] = np.shape(parameters)
      selector = selector.fit(parameters[:,:b-1], np.transpose(layer))
      selector.support_ 
      selector.ranking_
      relevance = selector.ranking_
      print(relevance)
      return relevance

def perf_pca(parameters):
# Performs a PCA, displays explained variance and space of the 3 first PC
      pca = PCA()
      [a,b] = np.shape(parameters)
      pca.fit(parameters[:,:b-1])
      var_ratio = pca.explained_variance_ratio_
      components = pca.transform(parameters[:,:b-1])
      return components, var_ratio

def GPC_classify(parameters, layer):
# Classification based on the selected parameters, based on Gaussian Process
     
      X = parameters      
      y = layer
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
      kernel = 1.0 * RBF(1.0)
      clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X_train, y_train)
      a = clf.predict(X_test)     
      score = accuracy_score(y_test,a)
      return score


def plot_confusion_matrix(parameters, layer, dimensions):
# Plots the confusion matrix



      cmap=plt.cm.Blues
      X = parameters[:,dimensions]      
      y = layer
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
      kernel = 1.0 * RBF(1.0)
      clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X_train, y_train)
      a = clf.predict(X_test)   
      score = accuracy_score(y_test,a)
      cm = confusion_matrix(y_test, a)
      fig, ax = plt.subplots()
      im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
      ax.figure.colorbar(im, ax=ax)
      ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

      plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
      fmt = 'd'
      thresh = cm.max() / 2.
      for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
      fig.tight_layout()
      return ax, score

def display_features(parameters,layer,features):
# After choosing 2 parameters, plots them + the result of training a GP on them

          plt.figure()
          ax = plt.subplot(111)
          
          X = parameters[:,features]

          h = .2
          y = np.squeeze(layer)


          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
          kernel = 1.0 * RBF(1.0)
          clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X_train, y_train)
          cm = plt.cm.RdBu
          cm_bright = ListedColormap(['#FF0000', '#0000FF'])
          ax.scatter(X[:,0],X[:,1],c=y, cmap=cm_bright,
               edgecolors='k')
          x_min, x_max = X[:, 0].min() - .5,X[:, 0].max() + .5
          y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
          xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
          np.arange(y_min, y_max, h))
      
          Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]      
        # Put the result into a color plot
          Z = Z.reshape(xx.shape)
          ax.contourf(xx, yy, Z, cmap=cm, alpha=.35)     
          a = clf.predict(X_test)          
          score = accuracy_score(y_test,a)
          ax.text(0, 0, ('%.2f' % score).lstrip('0'), size=15,horizontalalignment='right', verticalalignment='center')

          return
    
def perform_tsne(parameters, layer):
# Perform tSNE and use the resulting space to train a GP

      x = parameters
      y = np.squeeze(layer)
      yy = TSNE(n_components=2).fit_transform(x)
      cm = plt.cm.RdBu
      cm_bright = ListedColormap(['#FF0000', '#0000FF'])
      h = 0.2                            
      plt.figure()
      ax = plt.subplot(111)
      ax.scatter(yy[:,0],yy[:,1],c=y, cmap=cm_bright,
               edgecolors='k')
      X_train, X_test, y_train, y_test = train_test_split(yy, y, test_size=0.33)
      kernel = 1.0 * RBF(1.0)
      clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X_train, y_train)
      x_min, x_max = yy[:, 0].min() - .5,yy[:, 0].max() + .5
      y_min, y_max = yy[:, 1].min() - .5, yy[:, 1].max() + .5
      xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
      np.arange(y_min, y_max, h))
      
      Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]      
        # Put the result into a color plot
      Z = Z.reshape(xx.shape)
      ax.contourf(xx, yy, Z, cmap=cm, alpha=.35)        
      a = clf.predict(X_test)
    
      score = accuracy_score(y_test,a)
      ax.text(0, 0, ('%.2f' % score).lstrip('0'), size=15,horizontalalignment='right', verticalalignment='center')
      return