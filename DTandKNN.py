# -*- coding: utf-8 -*-
"""Fall2020_CS146_HW1.ipynb


# **Programming Excercise 1**

"""

import sys

# To add your own Drive Run this cell.
from google.colab import drive
drive.mount('/content/drive')

# Please append your own directory after ‘/content/drive/My Drive/'
# where you have nutil.py and adult_subsample.csv
### ========== TODO : START ========== ###

sys.path += ['/content/drive/My Drive/ECEM146'] 

### ========== TODO : END ========== ###

from nutil import *

# Use only the provided packages!
import math
import csv

from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

######################################################################
# Immutatble classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        n,d = X.shape
        y = [self.prediction_] * n
        return y

######################################################################
# Mutatble classes
######################################################################

class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = {}

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        neg = 0
        pos = 0

        for i in np.nditer(y):
          if i==0:
            neg = neg + 1
          elif i==1:
            pos = pos + 1

        temp = (float)(neg+pos)
        pneg = (float)((float)(neg)/temp)
        ppos = (float)((float)(pos)/temp)
        self.probabilities_[0] = pneg
        self.probabilities_[1]=ppos
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        y = np.random.choice(2,X.shape[0],p= [self.probabilities_[0],self.probabilities_[1]])
        #print("this is XXXXXXXX",y)
        ### ========== TODO : END ========== ###

        return y

######################################################################
# Immutatble functions
######################################################################

def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    ncol = 3
    nrow = d // ncol + 1
    for i in range(d) :
        fig.add_subplot (nrow,ncol,i+1)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels

######################################################################
# Mutatble functions
######################################################################

def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
        f1_score    -- float, test "micro" averaged f1 score
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error using StratifiedShuffleSplit over ntrials
    # hint: use train_test_split (be careful of the parameters)
    train_error = 0
    test_error = 0
    train_score=[]
    test_score=[]

    # tried to use StratifiedShuffleSplit but got array size error i couldnt figure out so used train_test_split instead.

    for i in range(ntrials):

      X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=test_size)
      clf.fit(X_train,y_train)

      y_test_pred = clf.predict(X_test)
      y_train_pred = clf.predict(X_train)
      temp1= 1-metrics.accuracy_score(y_train,y_train_pred, normalize = True)
      temp2 = 1-metrics.accuracy_score(y_test,y_test_pred, normalize = True)
      train_score.append(temp1)
      test_score.append(temp2)



    train_error = sum(train_score)/len(train_score)
    test_error = sum(test_score)/len(test_score)
    f1_score = metrics.f1_score(y_train,y_train_pred,average='micro');
   
    ### ========== TODO : END ========== ###

    return train_error, test_error, f1_score

######################################################################
# Immutatble functions
######################################################################


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()

######################################################################
# main
######################################################################

def main():
    
    
    
    # load adult_subsample dataset with correct file path
    ### ========== TODO : START ========== ###
    data_file =  "/content/drive/My Drive/ECEM146/adult_subsample.csv"
    ### ========== TODO : END ========== ###
    



    data = load_data(data_file, header=1, predict_col=-1)

    X = data.X; Xnames = data.Xnames
    y = data.y; yname = data.yname
    n,d = X.shape  # n = number of examples, d =  number of features

    

    #plt.figure()
    #========================================
    # part a: plot histograms of each feature
    #print('Plotting...')
    #plot_histograms (X, y, Xnames=Xnames, yname=yname)
    

    ### ========== TODO : START ========== ###
    # part i: Preprocess X (e.g., normalize)
    #norm = normalize(X)
    #X = norm

    ### ========== TODO : END ========== ###




    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)





    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    
    print('Classifying using Random...')
    clfR = RandomClassifier()
    clfR.fit(X,y)
    y_pred = clfR.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    ### ========== TODO : END ========== ###





    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    print('Classifying using Decision Tree...')
    clfDTC = DecisionTreeClassifier(criterion='entropy')
    clfDTC.fit(X,y)
    y_pred = clfDTC.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.6f' % train_error)
    ### ========== TODO : END ========== ###






    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors

    #K = 3
    print('Classifying using k-Nearest Neighbors...')
    KNN_3 = KNeighborsClassifier(n_neighbors=3)
    KNN_3.fit(X,y)
    y_pred = KNN_3.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.6f' % train_error)

    #K = 5
    KNN_5 = KNeighborsClassifier(n_neighbors=5)
    KNN_5.fit(X,y)
    y_pred = KNN_5.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.6f' % train_error)

    #K = 7
    KNN_7 = KNeighborsClassifier(n_neighbors=7)
    KNN_7.fit(X,y)
    y_pred = KNN_7.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.6f' % train_error)

    
    ### ========== TODO : END ========== ###





    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    train_error,test_error,f1_score = error(clf,X,y)
    print("Majority Vote Classifier")
    print('\t-- training error: %.6f' % train_error)
    print('\t-- test error: %.6f' % test_error)

    train_error,test_error,f1_score = error(clfR,X,y)
    print("Random")
    print('\t-- training error: %.6f' % train_error)
    print('\t-- test error: %.6f' % test_error)

    train_error,test_error,f1_score = error(clfDTC,X,y)
    print("Decision tree")
    print('\t-- training error: %.6f' % train_error)
    print('\t-- test error: %.6f' % test_error)

    train_error,test_error,f1_score = error(KNN_5,X,y)
    print("KNN")
    print('\t-- training error: %.6f' % train_error)
    print('\t-- test error: %.6f' % test_error)
    ### ========== TODO : END ========== ###





    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k...')
    """
    k = list(range(1,50,2)) #starts at 1 ends at 50 and step is 2 to have only odd numbers
    cross_val = []
    for i in k:
        clf = KNeighborsClassifier(n_neighbors=i)
        cross_val.append(1 - np.mean(cross_val_score(clf, X, y, cv = 10)) )
    plt.plot(k,cross_val)
    """
    ### ========== TODO : END ========== ###





    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    """
    d =list(range(1,21,1))
    train_score = []
    test_score = []
    for i in range(1, 21):
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=i)
        train_error, test_error,F1_score = error(clf, X, y)
        train_score.append(train_error)
        test_score.append(test_error)
    plt.figure()
    line, = plt.plot(d,train_score,color='blue')
    line1, = plt.plot(d,test_score,color = 'red')
    plt.legend((r'training',r'testing'))
    """

    ### ========== TODO : END ========== ###





    ### ========== TODO : START ========== ###
    # part h: investigate decision tree and k-Nearest Neighbors classifier with various training set sizes

    #Using train_test_split to split data into 10 and 90% and using 10% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0) 
    increment = int(len(X_train)/10)
    knn_train_error = []
    knn_test_error = []

    decision_train_error = []
    decision_test_error = []

    inr = np.array(range(1,11,1))

    for i in inr:
    trainingErr = 0
    testingErr = 0
    for j in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=j)
        clf = KNeighborsClassifier(n_neighbors=7)
        X1 = X_train[0 : i*increment ]
        y1 = y_train[0 : i*increment ]
        clf.fit(X1, y1)  # fit training data using the classifier
        train_pred = clf.predict(X1)  # take the classifier and run it on the training data
        test_pred = clf.predict(X_test)
        trainingErr += 1 - metrics.accuracy_score(y1, train_pred, normalize=True)
        testingErr += 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)

    trainingErr = trainingErr/ 100
    testingErr = testingErr/ 100
    knn_train_error.append(trainingErr)
    knn_test_error.append( testingErr )
    # print(testingErr)

for i in inr:
    trainingErr = 0
    testingErr = 0
    for j in range(100):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=j)
        clf = DecisionTreeClassifier(max_depth=3)
        X1 = X_train[0 : i*increment ]
        y1 = y_train[0 : i*increment ]
        clf.fit(X1, y1)  # fit training data using the classifier
        train_pred = clf.predict(X1)  # take the classifier and run it on the training data
        test_pred = clf.predict(X_test)
        trainingErr += 1 - metrics.accuracy_score(y1, train_pred, normalize=True)
        testingErr += 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)

    trainingErr = trainingErr/ 100
    testingErr = testingErr/ 100
    decision_train_error.append(trainingErr)
    decision_test_error.append( testingErr )

        # print(error_test)

    plt.plot((inr/10),knn_train_error)
    plt.plot((inr/10),knn_test_error)
    plt.plot((inr/10),decision_train_error)
    plt.plot((inr/10),decision_test_error)
    plt.legend((r'knn training',r'knn testing',r'decision tree training',r'decision tree testing'))
    
    ### ========== TODO : END ========== ###



    print('Done')


if __name__ == "__main__":
    main()

"""# New Section"""
