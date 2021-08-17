"""
@author: %(Javier Alegre Cortés)s
@email: %(javier.alegre@hotmail.es)
@institution: %(Instituto de Neurociencias (UMH-CSIC))
"""
#%%
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
import matplotlib.pyplot as plt
import numpy as np
import core_functions
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1006, 681)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(330, 20, 191, 81))
        font = QtGui.QFont()
        font.setFamily("Impact")
        font.setPointSize(28)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 270, 191, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 240, 141, 23))
        self.pushButton.setObjectName("pushButton")
        self.statsBtn = QtWidgets.QPushButton(self.centralwidget)
        self.statsBtn.setGeometry(QtCore.QRect(610, 80, 81, 31))
        self.statsBtn.setObjectName("statsBtn")
        self.layerBtn = QtWidgets.QPushButton(self.centralwidget)
        self.layerBtn.setGeometry(QtCore.QRect(610, 120, 161, 31))
        self.layerBtn.setObjectName("layerBtn")
        self.classifyBtn = QtWidgets.QPushButton(self.centralwidget)
        self.classifyBtn.setGeometry(QtCore.QRect(780, 80, 161, 31))
        self.classifyBtn.setObjectName("classifyBtn")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 0, 221, 211))
        self.label_2.setObjectName("label_2")
        self.DimEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.DimEdit.setGeometry(QtCore.QRect(10, 410, 191, 20))
        self.DimEdit.setObjectName("DimEdit")
        self.TraceEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.TraceEdit.setGeometry(QtCore.QRect(10, 500, 191, 20))
        self.TraceEdit.setObjectName("TraceEdit")
        self.traceBtn = QtWidgets.QPushButton(self.centralwidget)
        self.traceBtn.setGeometry(QtCore.QRect(780, 120, 161, 31))
        self.traceBtn.setObjectName("traceBtn")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 390, 291, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 470, 221, 20))
        self.label_4.setObjectName("label_4")
        self.varAxis = QtWidgets.QGraphicsView(self.centralwidget)
        self.varAxis.setGeometry(QtCore.QRect(270, 200, 721, 441))
        self.varAxis.setObjectName("varAxis")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 310, 291, 21))
        self.label_5.setObjectName("label_5")
        self.featEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.featEdit.setGeometry(QtCore.QRect(10, 340, 191, 20))
        self.featEdit.setObjectName("featEdit")
        self.featBtn = QtWidgets.QPushButton(self.centralwidget)
        self.featBtn.setGeometry(QtCore.QRect(110, 310, 75, 23))
        self.featBtn.setObjectName("featBtn")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(695, 80, 75, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(280, 110, 301, 16))
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(10, 0, 1006, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton.clicked.connect(self.load_data_fcn)
        self.layerBtn.clicked.connect(self.rfe_fcn)
        self.statsBtn.clicked.connect(self.pca_fcn)
        pixmap = QPixmap('bitmap.png')
        pixmap = pixmap.scaled(self.label_2.width(),self.label_2.height(),QtCore.Qt.KeepAspectRatio)
        self.label_2.setPixmap(pixmap)
        self.classifyBtn.clicked.connect(self.clasify_fcn)
        self.traceBtn.clicked.connect(self.plt_trace_fcn)
        self.featBtn.clicked.connect(self.plot_features_fcn)
        self.pushButton_2.clicked.connect(self.tsne_fcn)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "NekoStats"))
        self.label.setText(_translate("MainWindow", "Rest&Test"))
        self.pushButton.setText(_translate("MainWindow", "Load data"))
        self.statsBtn.setText(_translate("MainWindow", "PCA"))
        self.layerBtn.setText(_translate("MainWindow", "Recursive Feature Extraction"))
        self.classifyBtn.setText(_translate("MainWindow", "Classify"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.TraceEdit.setText(_translate("MainWindow", "1"))
        self.traceBtn.setText(_translate("MainWindow", "Plot Trace"))
        self.label_3.setText(_translate("MainWindow", "Dimensions to use for classification"))
        self.label_4.setText(_translate("MainWindow", "Trace(s) to plot"))
        self.label_5.setText(_translate("MainWindow", "Features to plot"))
        self.featBtn.setText(_translate("MainWindow", "Plot"))
        self.pushButton_2.setText(_translate("MainWindow", "tSNE"))
        self.label_6.setText(_translate("MainWindow", "Feature extraction and supervised classification of time series"))


    def load_data_fcn(self, btn):
          global dataset, fileName, parameters, layer
          dataset, parameters, fileName, layer = core_functions.load_data()
          self.lineEdit.setText(fileName)
          # parameters = core_functions.compute_parameters(dataset, layer)
          plt.figure()
          manager = plt.get_current_fig_manager()
          manager.window.showMaximized()
          [a,b] = np.shape(parameters)
          for i in range(b):
              for j in np.unique(layer[0]):
                ax = plt.subplot(4,4,i+1)
                plt.ylabel('Feature ' + str(i))
                ax.violinplot(parameters[np.where(layer[0]==j)[0],i],[j])
          return dataset, parameters
    
      
    def rfe_fcn(self, btn):
            from mpl_toolkits.mplot3d import Axes3D
            global relevance
            relevance = core_functions.rfe(parameters, layer)
            relevance = np.where(relevance==1)[0]
            score = core_functions.GPC_classify(parameters[:,relevance],np.squeeze(layer))                     
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(parameters[np.where(layer[0]==1)[0],relevance[0]],parameters[np.where(layer[0]==1)[0],relevance[1]],parameters[np.where(layer[0]==1)[0],relevance[2]])
            ax.scatter(parameters[np.where(layer[0]==2)[0],relevance[0]],parameters[np.where(layer[0]==2)[0],relevance[1]],parameters[np.where(layer[0]==2)[0],relevance[2]])
            ax.text(0, 0, 0, ('%.2f' % score).lstrip('0'), size=15,horizontalalignment='right', verticalalignment='center')
            plt.xlabel('Feature ' + str(relevance[0]))
            plt.ylabel('Feature ' + str(relevance[1]))
            plt.zlabel('Feature ' + str(relevance[2]))
    def pca_fcn(self, btn):
            from mpl_toolkits.mplot3d import Axes3D
            global components, var_ratio
            components, var_ratio = core_functions.perf_pca(parameters) 
            score = core_functions.GPC_classify(components,np.squeeze(layer))                     
            plt.figure()
            plt.bar(np.arange(np.size(var_ratio)),var_ratio*100)
            plt.title('Explained variance')
            plt.xlabel('Component')
            plt.ylabel('% of explained variance')
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(components[np.where(layer[0]==1)[0],[0]],components[np.where(layer[0]==1)[0],[1]],components[np.where(layer[0]==1)[0],[2]])
            ax.scatter(components[np.where(layer[0]==2)[0],[0]],components[np.where(layer[0]==2)[0],[1]],components[np.where(layer[0]==2)[0],[2]])
            ax.text(0, 0, 0, ('%.2f' % score).lstrip('0'), size=15,horizontalalignment='right', verticalalignment='center')
            ax.set_xlabel('Principal component 1')
            ax.set_ylabel('Principal component 2')
            ax.set_zlabel('Principal component 3')

    def clasify_fcn(self, btn):
            global dimensions
            print(self.DimEdit.text())
            dimensions = str(self.DimEdit.text())
            dimensions = [int(x) for x in dimensions.split(' ')]
            dimensions = np.array(dimensions)
            ax, score = core_functions.plot_confusion_matrix(parameters, np.squeeze(layer), dimensions)
            print(score)
            return dimensions                   
          
                                         
    def plt_trace_fcn(self, btn):
            global trace
            print(self.TraceEdit.text())
            trace = str(self.TraceEdit.text())
            trace = [int(x) for x in trace.split(' ')]
            trace = np.array(trace)
            plt.figure()
            for i in range(np.size(trace)):
                  plt.subplot(np.size(trace),1, i+1)
                  plt.plot(dataset[trace[i],:])
                  plt.ylabel('Trace ' + str(i))
            return trace
    

    def plot_features_fcn(self, btn):
          global features
          features = str(self.featEdit.text())
          features = [int(x) for x in features.split(' ')]
          features = np.array(features) 
          core_functions.display_features(parameters,layer,features)
# =============================================================================
#           self.varAxis.plot(parameters[layer==1,features[0]],parameters[layer==1,features[1]],'ob')
#           self.varAxis.plot(parameters[layer==2,features[0]],parameters[layer==2,features[1]],'or')
#           self.draw()
# =============================================================================
          return features

    def tsne_fcn(self,btn):
          core_functions.perform_tsne(parameters,layer)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

