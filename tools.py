# 2014.05.27 13:49:52 PKT
"""
Copyright (c) 2014-15, Sibt ul Hussain <sibt.ul.hussain at gmail dot com>
All rights reserved.
Released under BSD License
-------------------------
For license terms please see license.lic

----------------------------

Set of utility functions.
"""
from struct import *
import argparse
import numpy as np
import cPickle as cp
import time
from scipy.io import loadmat
from scipy.io import savemat
import os
import sys
from shutil import *
import shutil
# from uniconvertor.app.utils.os_utils import get_files_withpath, get_files

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')



def check_values(r, coptions):
    """ Validate given input values for the valid values """
    if r not in coptions:
        raise ValueError('Wrong option' + r + ', choose among the following options: ' + ' '.join(coptions))



def check_islist(val):
    """ convert a scalar to a list, if the input is already a list returns it as it is """
    if type(val) == list:
        return val
    return [val]



def str2list(r):
    """ Return list of values by tokenizing input string list"""
    if isinstance(r, list) or type(r) == int or type(r) == float:
        return r
    else:
        if r.find('[') != -1:
            if r.find(',') != -1:
                values = map(int, r.strip('[').strip(']').split(','))
            else:
                values = map(int, r.strip('[').strip(']').split())
        else:
            values = int(r)
        return values



def list2file(fname, imlist, odir=''):
    """ Write a list into a list file """
    ofile = open(fname, 'w')
    for im in imlist:
        if len(odir) != 0:
            im = os.path.join(odir, os.path.basename(im))
        ofile.write(im + '\n')

    ofile.close()



def read_txt_file(fname):
    """ read a txt file, each line is read as a separate element in the returned list"""
    return open(fname).read().splitlines()



def get_number_lines(file):
    """
        return numbers of lines in a file...
       
    """
    return len(read_txt_file(file))



def get_dirs(path, pattern=""):
    """ return path of (sub-)directories present at the given path """
    check_path(path)
    dirs = []
    for n in os.listdir(path):
        dpath = os.path.join(path, n)
        if os.path.isdir(dpath) :
            if len(pattern) == 0:
                dirs.append(dpath)
            elif dpath.lower().count(pattern):
                dirs.append(dpath)
    if len(dirs) > 0:
        return dirs



def get_dirs_withpath(path='.'):
    """ return lists of directories with full path         
    """
    list = []
    names = []
    if os.path.isdir(path):
        try:
            names = os.listdir(path)
        except os.error:
            return names
    names.sort()
    for name in names:
        if os.path.isdir(os.path.join(path, name)) and name != '.svn' and name != '.git':
            list.append(os.path.join(path, name))

    return list



def getfiles(path, types):
    """Get list of files available at a given path
      types: a list of possible files to extract, it can be any type.
      Example: getfiles('/tmp/',['.txt','.cpp','.m']); 
    
    """
    check_path(path)
    imlist = []
    for filename in os.listdir(path):
        if os.path.splitext(filename)[1].lower() in types:
            imlist.append(os.path.join(path, filename))

    return imlist



def create_directory(dir):
    """ Create a new directory recursively
    Note that directory name must note contain any spaces, otherwise it will generate errors"""
    if os.path.isdir(dir):
        return 
    parent, base = os.path.split(dir)
    if len(parent) == 0 and len(base) != 0:
        os.mkdir(base, 511)
        return 
    create_directory(parent)
    os.mkdir(dir, 511)



def check_path(fname, message=''):
    """ Function check for the validity of path (existence of file or directory path),
    if not found raise exception"""
    if len(message) == 0:
        message = 'path ' + fname + ' Not found'
    if not os.path.exists(fname):
        print message
        raise ValueError(message)

def remove_spaces_dirs(path, lchars):
    """ Function remove spaces and  characters (lchars)
    from all the subdirectories
    of the given path with the corresponding characters,
    e.g. remove_space_dirs('./',[(' ',''), ( '(' , '-' ), ( ')','')])
    will replace all the spaces, replaces '(' with '-' and so on  
    """
    sdirs = get_dirs('.')
    print sdirs
    if sdirs is None or len(sdirs) == 0:
        return 
    for d in sdirs:
        os.chdir(d)
        remove_spaces_dirs('./', lchars)
        os.chdir(os.path.pardir)
        nd = d
        for c in lchars:
            nd = nd.replace(c[0], c[1])

        # os.rename(d, nd)
        shutil.move(d, nd)
        
def generate_valid_filename(fname):
    '''
        function removes punctuations and special symbols from a given file name...
        
    '''
    return "".join(x for x in fname if x.isalnum())

# Return file list for provided path
# copied from os_utils...

def get_files(path='.', ext='*',withpath=False):    

    """
    Returns the list of files with extension (ext) at the given path, withpath appended or not...

    Input:
    -----------
    path: path of the directory
    ext: extension to search for at the given path
    withpath: whether to append the path or not...

    Returns:
    ------------
    list of files found with the extension ext at the given path

    """
    flist = []

    if path:
        if os.path.isdir(path):
            try:
                names = os.listdir(path)
            except os.error:
                return []
        names.sort()
        for name in names:
            if not os.path.isdir(os.path.join(path, name)):
                if ext == '*':
                    flist.append(name)
                elif '.' + ext == name[-1 * (len(ext) + 1):]:
                    flist.append(name)              

    if withpath:
        return [os.path.join(path,fname) for fname in flist]
    else:
        return flist


def parse_args(argv=None):
    """ Parse command line arguments and do the processing"""
    if argv == None:
        argv = sys.argv
    print 'ARGV:',
    print argv[1:]
    if len(sys.argv) == 1 or len(sys.argv) < 4:
        print ' Invalid Arguments: Usage tools.py <zip_file_name> <assignment_name> <destination_folder> \n'
        print 'E.g., tools.py bulk_download.zip assignment_1 /tmp/'
        sys.exit(1)
    zfile = argv[1]
    aname = argv[2]
    dfolder = argv[3]
    tmpdir = dfolder + aname
    create_directory(tmpdir)
    uzip = 'unzip ' + zfile + ' -d ' + tmpdir + os.path.sep
    print ' Unzipping.... ' + uzip
    os.system(uzip)
    os.chdir(tmpdir)
    # remove spaces from directories
    remove_spaces_dirs('./', [(' ', ''), ('(', '-'), (')', '')])
    d = get_dirs('./')
    os.chdir(d[0])
    # get list of student directores
    d = get_dirs('./')
    for f in d:
        print f
        if f[2] == 'i' and f[3].isdigit():
            nd = get_dirs('./' + f + os.path.sep, 'submission')[0]
            nname = f.partition(',')[0][2:]
            # os.rename(nd, nname)
            shutil.move(nd, nname)
            # check for .zip file
            sdir = './' + os.path.sep + nname + os.path.sep  # sub directory
            zfile = get_files(sdir, 'zip')
            if len(zfile) != 0:  # unzip the contents in the folder...
                os.chdir(sdir)
                tfile = generate_valid_filename(zfile[0])
                shutil.move(zfile[0], tfile)
                os.system('unzip -j ' + tfile)
                os.chdir(os.path.pardir)
            zfile = get_files(sdir, 'rar')
            if len(zfile) != 0:  # unrar the contents in the folder...
                os.chdir(sdir)
                tfile = generate_valid_filename(zfile[0])
                shutil.move(zfile[0], tfile)
                os.system('unrar e ' + generate_valid_filename(zfile[0]))
                os.chdir(os.path.pardir)
            
            zfile = get_files(sdir)
            if len(zfile) == 0:
                # os.removedirs(sdir)
                shutil.rmtree(sdir, ignore_errors=True)
        # os.removedirs(f)
        shutil.rmtree(f, ignore_errors=True)
    

#http://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals/12321306#12321306

def plot_cov_ellipse_by_volume(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """

    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)
    
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    Usage
    -------
    

    # Generate some random, correlated data
    points = np.random.multivariate_normal(
            mean=(1,1), cov=[[0.4, 9],[9, 10]], size=1000
            )
    # Plot the raw points...
    x, y = points.T
    plt.plot(x, y, 'ro')

    # Plot a transparent 3 standard deviation covariance ellipse
    plot_point_cov(points, nstd=3, alpha=0.5, color='green')
    
    """
    import numpy as np
   
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
    
#Training utilities...
def plotCov(X,Y,colors=[]): 
    """ Function plots the covariance matrix for each class ... 
        for the given data...
    Input
    ---------
        X: M x n training matrix for K classes
        Y: M x 1 label matrix
        colors: a list of string with 'K' distinct colors...
    
    """ 
    import matplotlib.pyplot as plt
    nexamples, nfeatures=X.shape
    classes=np.unique(Y) # extract different classes...        
    
    if len(colors)==0:
        colors=['red','green','blue','black','orange']
    
    markers=['ro','gs','b^']
    for c, klass in enumerate(classes):
        # find the index of each class
        idx= Y==klass        
        plt.plot(X[idx,0], X[idx,1], markers[c])
        plot_point_cov(X[idx,:], nstd=3, alpha=0.4, color=colors[c])
    plt.xlim([0, 9])
    plt.ylim([0, 3])
    plt.legend(classes)

def generate_folds(X, Y, nfolds=4):
    """
    Split the training data into startified-nfolds... 
    
    Parameters:
    --------
    X: training examples
    Y: training labels
    nfolds: number of folds...
        By default split according to fix 75-25% ratio
    
    Returns:
    ---------
    list of lists, with each nested list containing four elements
    training data, training labels, test data, test labels per fold
    """
    classes = np.unique(Y)
    print 'Generating CV data for {} classes'.format(len(classes))
    
    cvlist = []  # four elements per nested-list
    # idxlist=[] # 2 elements per list to contain indeces...
    
    for cidx, c in enumerate(classes):
        idx = Y == c  # find class example indeces
        Yt = Y[idx]  # get class labels
        Xt = X[idx, :]  # get training examples
        nexamples = Xt.shape[0]
        # Generate a random permutation of the indeces
        ridx = np.arange(nexamples)  # generate indeces
        np.random.shuffle(ridx)  # shuffle randomly ...

        # number of test examples per fold
        nexamples = nexamples / nfolds
        sridx = set(ridx)  # create a set of all indeces
        sidx = 0
        for k in range(nfolds):
            testidx = ridx[sidx:sidx + nexamples]
            trainidx = list(sridx.difference(testidx)) # take a set difference
            sidx += nexamples
            
            if cidx == 0:
                cvlist.append([Xt[trainidx, :], Yt[trainidx], Xt[testidx, :], Yt[testidx]])
                #idxlist.append([trainidx, testidx])
#                 cvlist[k][0]=Xt[trainidx,:]
#                 cvlist[k][1]=Yt[trainidx]
#                 cvlist[k][2]=Xt[testidx,:]
#                 cvlist[k][3]=Yt[testidx]
            else:  # append to the set...
                # pdb.set_trace()
                cvlist[k][0] = np.vstack((cvlist[k][0], Xt[trainidx, :]))
                cvlist[k][1] = np.hstack((cvlist[k][1], Yt[trainidx]))
                cvlist[k][2] = np.vstack((cvlist[k][2], Xt[testidx, :]))
                cvlist[k][3] = np.hstack((cvlist[k][3], Yt[testidx]))
                
                #idxlist[k][0]=np.hstack((idxlist[k][0],trainidx))
                #idxlist[k][1]=np.hstack((idxlist[k][1],testidx))
            # print cidx, k, cvlist[k][0].shape, cvlist[k][2].shape
    return cvlist

def split_data(X, Y, percentage=0.7):
    """
     Split the training data into training and test set according to given percentage... 
    
    Parameters:
    --------
    X: training examples
    Y: training labels
    percentage: split data into train and test accorind to given %
    
    Returns:
    ---------    
    returns four lists as tuple: training data, training labels, test data, test labels 
    """
    
    testp=1-percentage

    #Split the data into train and test according to given fraction..

    #Creat a list of tuples according to the n-classes where each tuple will 
    # contain the pair of training and test examples for that class...
    #each tuple=(training-examples, training-labels,testing-examples,testing-labels)
    exdata=[]
    #Creat 4 different lists 
    traindata=[]
    trainlabels=[]
    testdata=[]
    testlabels=[]

    classes=np.unique(Y)

    for c in classes:
        # print c
        idx=Y==c
        Yt=Y[idx]
        Xt=X[idx,:]
        nexamples=Xt.shape[0]
        # Generate a random permutation of the indeces
        ridx=np.arange(nexamples) # generate indeces
        np.random.shuffle(ridx)
        ntrainex=round(nexamples*percentage)
        ntestex=nexamples-ntrainex

        traindata.append(Xt[ridx[:ntrainex],:])
        trainlabels.append(Yt[ridx[:ntrainex]])

        testdata.append(Xt[ridx[ntrainex:],:])
        testlabels.append(Yt[ridx[ntrainex:]])

        #exdata.append((Xt[ridx[:ntrainex],:], Yt[ridx[:ntrainex]], Xt[ridx[ntrainex:],:], Yt[ridx[ntrainex:]]))


    # print traindata,trainlabels
    Xtrain=np.concatenate(traindata)
    Ytrain=np.concatenate(trainlabels)
    Xtest=np.concatenate(testdata)
    Ytest=np.concatenate(testlabels)
    return Xtrain, Ytrain, Xtest, Ytest

def plot_decision_regions(X, y, clf, res=0.02, cycle_marker=True, legend=1):
    """
    Plots decision regions of a classifier.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
      Feature Matrix.

    y : array-like, shape = [n_samples]
      True class labels.

    clf : Classifier object. Must have a .predict method.

    res : float (default: 0.02)
      Grid width. Lower values increase the resolution but
        slow down the plotting.

    cycle_marker : bool
      Use different marker for each class.

    legend : int
      Integer to specify the legend location.
      No legend if legend is 0.

    cmap : Custom colormap object.
      Uses matplotlib.cm.rainbow if None.

    Returns
    ---------
    None

    Examples
    --------

    from sklearn import datasets
    from sklearn.svm import SVC

    iris = datasets.load_iris()
    X = iris.data[:, [0,2]]
    y = iris.target

    svm = SVC(C=1.0, kernel='linear')
    svm.fit(X,y)

    plot_decision_region(X, y, clf=svm, res=0.02, cycle_marker=True, legend=1)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('SVM on Iris')
    plt.show()
    # Sebastian Raschka 08/13/2014
    # mlxtend Machine Learning Library Extensions
    # matplotlib utilities for removing chartchunk

    """
    from itertools import cycle
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    if type(y[0])==np.string_ or type(y[0])==str:
        # Map to integer labels...
#       print " Hello"
        nclasses=list(np.unique(y));
        y=np.array([nclasses.index(label) for label in y])
    
    
    
    marker_gen = cycle('sxo^v')

    # make color map
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    classes = np.unique(y)
    n_classes = len(np.unique(y))
    if n_classes > 5:
        raise NotImplementedError('Does not support more than 5 classes.')
    cmap = matplotlib.colors.ListedColormap(colors[:n_classes])

    # plot the decision surface

    # 2d
    if len(X.shape) == 2 and X.shape[1] > 1:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 1D
    else:
        y_min, y_max = -1, 1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    # 2d
    if len(X.shape) == 2 and X.shape[1] > 1:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    # 1D
    else:
        y_min, y_max = -1, 1
        Z = clf.predict(np.array([xx.ravel()]).T)

    
    if type(Z[0])==np.string_:
        # Map to integer labels...
        # print " Hello"
        nclasses=list(np.unique(Z));
        Z=np.array([nclasses.index(label) for label in Z])
    
    
    
    
    print Z, type(Z[0]), type(Z[0])==np.string_
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples
    for c in np.unique(y):

        if len(X.shape) == 2 and X.shape[1] > 1:
            dim = X[y==c, 1]
        else:
            dim = [0 for i in X[y==c]]

        plt.scatter(X[y==c, 0],
                    dim,
                    alpha=0.8,
                    c=cmap(c),
                    marker=next(marker_gen),
                    label=c)

    if legend:
        plt.legend(loc=legend, fancybox=True, framealpha=0.5)




def powerset(iterable):
    """
    Computes the powerset (all possible 2^n subsets of given set).

    Parameters
    ----------
    iterable : an iterable element (like a list, set)

    Returns:
    ----------
    an itertor over the poweret to avoid memory overhead.

    Example
    ---------
    y=powerset(['A','B','C'])
    print set(y)
    will result in following output:
    set([('B', 'C'), ('A',), ('C',), ('B',), (), ('A', 'B', 'C'), ('A', 'B'), ('A', 'C')])

    """
    from itertools import chain, combinations
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) )

def get_powerset(iterable, length):
    """
    Computes the powerset (all possible 2^n subsets of given set) and returns 
    the subset with cardinality <= length.

    Parameters
    ----------
    iterable : an iterable element (like a list, set)

    Returns:
    ----------     
    a set of subsets with all subsets having cardinality <= length

    Example
    ---------
    y=get_powerset(['A','B','C'],2)
    print set(y)
    will result in following output:
    set([('B', 'C'), ('A',), ('C',), ('B',), ('A', 'B'), ('A', 'C')])

    """
    subsets=set(powerset(iterable))
    #create a set with each element another set
    ss=set([frozenset(s) for s in subsets if len(s)>=1 and len(s)<=length])
    # #create a set with each element a tuple
    # t=set([s for s in subsets if len(s)>=1 and len(s)<=length])
    return ss
if __name__ == '__main__':
    sys.exit(parse_args())


def print_confusion_matrix(plabels,tlabels):
    """
        functions print the confusion matrix for the different classes
        to find the error...
        
        Input:
        -----------
        plabels: predicted labels for the classes...
        tlabels: true labels for the classes
        
        code from: http://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python
    """
    import pandas as pd
    plabels = pd.Series(plabels)
    tlabels = pd.Series(tlabels)
    
    # draw a cross tabulation...
    df_confusion = pd.crosstab(tlabels,plabels, rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    #print df_confusion
    return df_confusion
