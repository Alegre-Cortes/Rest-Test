# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:07:30 2019

@author: ramon.reig
"""
from scipy.signal import find_peaks 
from scipy.stats import zscore
import numpy as np
def load_data():
#def load_data(self, btn):
#    global dataset, fileName
    print('Hello word')
    from PyQt5 import QtCore, QtGui, QtWidgets
    # import scipy.io as sio
    options = QtWidgets.QFileDialog.Options() 
    options |= QtWidgets.QFileDialog.DontUseNativeDialog 
    fileName,_ = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()","", options=options)
    fileName
    aux = np.load(fileName,allow_pickle=True)
    aux = aux.item()
    layer = aux['Labels']
    if ('Data' in aux):
        dataset = aux['Data']
        parameters = compute_parameters(dataset, layer)
    elif ('Parameters' in aux):
        parameters = aux['Parameters']
        dataset = 'No dataset was loaded'
    print(np.shape(dataset))

    return dataset, parameters, fileName, layer


def smooth(x,window_len=11,window='flat'):
    """smooth the data using a window with requested size.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    """

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def ComputeD2USpeed_ind_UP(Transition):
      TimeWindow = np.arange(np.size(Transition))+1
      S,p = np.polyfit(TimeWindow,Transition,1)
      Slope = S
      return Slope*20000


def compute_parameters(dataset, layer):
#      global dataset, parameters
      cell = []
      features = []
      x = dataset
      for num in range(np.squeeze(np.shape(x[:,0]))):
            
             a = smooth(x[num,:], window_len=200)
             a = a[99:]
             a = np.transpose(a)
             UD = []
             D2U =[]
             U2D = []
             partition = np.int(np.round(np.size(a)/5))
             for i in range(0,np.size(a)-partition,partition):
                   temp = a[i:i+partition]
                   thresh = np.mean(temp)+(0.5*np.std(temp))
                   temp_UD = np.zeros(np.size(temp))
                   for j in range(np.size(temp)):
                         if temp[j]>=thresh:
                               temp_UD[j] = 1
                   UD.extend(temp_UD)
             UD = np.asarray(UD)
             for i in range(np.size(UD)-1):
                  if (UD[i]==0 and UD[i+1]==1):
                        D2U.append(i)
                  elif (UD[i]==1 and UD[i+1]==0):
                        U2D.append(i)
             D2U = np.asarray(D2U)
             U2D = np.asarray(U2D)
             for i in range(1,np.size(U2D)):
                  temp = D2U[D2U>U2D[i]]
                  if (np.size(temp)>0 and temp[0] - U2D[i]<250):
                        UD[U2D[i]-1:temp[0]+1] = 1
                  
             D2U = []
             U2D = []
             for i in range(np.size(UD)-1):
                  if (UD[i]==0 and UD[i+1]==1):
                        D2U.append(i)
                  elif (UD[i]==1 and UD[i+1]==0):
                        U2D.append(i)
             D2U = np.asarray(D2U)
             U2D = np.asarray(U2D)
             for i in range(np.size(D2U)):
                  temp = U2D[U2D>[D2U[i]]]
                  if (np.size(temp)>0 and temp[0]-D2U[i]<200):
                        UD[D2U[i]:temp[0]+1] = 0
            
             D2U = []
             U2D = []
             for i in range(np.size(UD)-1):
                  if (UD[i]==0 and UD[i+1]==1):
                        D2U.append(i)
                  elif (UD[i]==1 and UD[i+1]==0):
                        U2D.append(i)
      
             D2U = np.asarray(D2U)
             U2D = np.asarray(U2D)                  
             for i in range(1,np.size(D2U)-1):
                   temp = U2D[U2D>D2U[i]]   
                   if (np.size(temp)>0 and temp[0] -D2U[i]>200):
                        Up = x[num,D2U[i]:temp[0]]
                        trans = []
                        trans.append(np.mean(Up))
                        trans.append(np.std(Up))
                        trans.append(min(Up[50:np.size(Up)-50]))
                        trans.append(np.max(Up))
                        trans.append(abs(np.max(Up))-abs(np.min(Up)))
                        trans.append(np.max(np.diff(smooth(Up,10))))
                        trans.append(np.min(np.diff(smooth(Up,10))))
                        peaks = find_peaks(zscore(smooth(Up, window_len=200)), distance=160, height=.3)
                        trans.append(np.size(peaks[0]))
                        trans.append(np.size(Up))
                        trans.append(np.mean(Up)-np.mean(x[num,D2U[i]-400:D2U[i]-200]))
                        trans.append(ComputeD2USpeed_ind_UP(x[num,D2U[i]-80:D2U[i]+50]))
                        trans.append(ComputeD2USpeed_ind_UP(x[num,U2D[i]:U2D[i]+120]))
                        trans.append(trans[10]/trans[11])
                        trans.append(layer[0,num])
                        trans = np.asarray(trans)
                        features.append(np.transpose(trans))
                        cell.append(num)
      features = np.asarray(features)
      #features = np.transpose(features)
      #%%
      features = np.concatenate((zscore(features[:,0:np.size(features[0,])-1], axis=0),np.expand_dims(features[:,np.size(features[0,])-1],-1)), axis=1)
      #features = np.transpose(features)
      
      #%%
      
      parameters = np.zeros((np.size(np.unique(cell)),14))
      for i in range(np.size(np.unique(cell))):                  
            for j in range(14):
                  index = [k for (k,val) in enumerate(cell) if val==i]
                  parameters[i,j] = np.mean(features[index,j])

      return parameters                    

       

def rfe(parameters, layer):
      from sklearn.feature_selection import RFE
      from sklearn.svm import SVC
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
      from sklearn.decomposition import PCA
      pca = PCA()
      [a,b] = np.shape(parameters)
      pca.fit(parameters[:,:b-1])
      var_ratio = pca.explained_variance_ratio_
      components = pca.transform(parameters[:,:b-1])
      return components, var_ratio

def GPC_classify(parameters, layer):
      from sklearn.model_selection import train_test_split
      from sklearn.gaussian_process import GaussianProcessClassifier
      from sklearn.metrics import accuracy_score
      from sklearn.gaussian_process.kernels import RBF, DotProduct
      from sklearn.metrics import accuracy_score
      
      X = parameters      
      y = layer
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
      kernel = 1.0 * RBF(1.0)
      clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X_train, y_train)
      a = clf.predict(X_test)     
      score = accuracy_score(y_test,a)
      return score


def plot_confusion_matrix(parameters, layer, dimensions):
      from sklearn.model_selection import train_test_split
      from sklearn.gaussian_process import GaussianProcessClassifier
      from sklearn.metrics import accuracy_score
      from sklearn.metrics import confusion_matrix
      from sklearn.gaussian_process.kernels import RBF, DotProduct
      from sklearn.metrics import accuracy_score


      cmap=plt.cm.Blues
      X = parameters[:,dimensions]      
      y = layer
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
      kernel = 1.0 * RBF(1.0)
      clf = GaussianProcessClassifier(kernel=kernel, warm_start=True).fit(X_train, y_train)
      a = clf.predict(X_test)   
      score = accuracy_score(y_test,a)

      cm = confusion_matrix(y_test, a)
    # Only use the labels that appear in the data


      fig, ax = plt.subplots()
      im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
      ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
      ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
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

          from sklearn.model_selection import train_test_split
          from sklearn.gaussian_process import GaussianProcessClassifier
          from sklearn.gaussian_process.kernels import RBF, DotProduct
          from sklearn.metrics import accuracy_score
          from matplotlib.colors import ListedColormap
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
      from matplotlib.colors import ListedColormap


      from sklearn.manifold import TSNE
      from sklearn.model_selection import train_test_split
      from sklearn.gaussian_process import GaussianProcessClassifier
      from sklearn.gaussian_process.kernels import RBF, DotProduct
      from sklearn.metrics import accuracy_score
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