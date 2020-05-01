import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from IPython.display import display
# import additional libraries for keras
import keras
from keras.utils.np_utils import to_categorical

# from keras.preprocessing.text import Tokenizer
from keras import models, layers, optimizers


from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam


def star_signals(signal, label_col=None, classes=None, 
                 class_names=None, figsize=(15,5), y_units=None, x_units=None):
    """
    Plots a scatter plot and line plot of time series signal values.  
    
    **ARGS
    signal: pandas series or numpy array
    label_col: name of the label column if using labeled pandas series
        -use default None for numpy array or unlabeled series.
        -this is simply for customizing plot Title to include classification    
    classes: (optional- req labeled data) tuple if binary, array if multiclass
    class_names: tuple or array of strings denoting what the classes mean
    figsize: size of the figures (default = (15,5))
    ******
    
    Ex1: Labeled timeseries passing 1st row of pandas dataframe
    > first create the signal:
    star_signal_alpha = train.iloc[0, :]
    > then plot:
    star_signals(star_signal_alpha, label_col='LABEL',classes=[1,2], 
                 class_names=['No Planet', 'Planet']), figsize=(15,5))
    
    
    Ex2: numpy array without any labels
    > first create the signal:
    
    >then plot:
    star_signals(signal, figsize=(15,5))
    """
    
    # pass None to label_col if unlabeled data, creates generic title
    if label_col is None:
        label = None
        title_scatter = "Scatterplot of Star Flux Signals"
        title_line = "Line Plot of Star Flux Signals"
        color='black'
        
    # store target column as variable 
    elif label_col is not None:
        label = signal[label_col]
        # for labeled timeseries
        if label == 1:
            cls = classes[0]
            cn = class_names[0]
            color='red'

        elif label == 2:
            cls = classes[1]
            cn = class_names[1] 
            color='blue'
    #create appropriate title acc to class_names    
        title_scatter = f"Scatterplot for Star Flux Signal: {cn}"
        title_line = f"Line Plot for Star Flux Signal: {cn}"
    
    # Set x and y axis labels according to units
    # if the units are unknown, we will default to "Flux"
    if y_units == None:
        y_units = 'Flux'
    else:
        y_units = y_units
    # it is assumed this is a timeseries, default to "time"   
    if x_units == None:
        x_units = 'Time'
    else:
        x_units = x_units
    
    # Scatter Plot 
    
    plt.figure(figsize=figsize)
    plt.scatter(pd.Series([i for i in range(1, len(signal))]), 
                signal[1:], marker=4, color=color, alpha=0.7)
    plt.ylabel(y_units)
    plt.xlabel(x_units)
    plt.title(title_scatter)
    plt.show();

    # Line Plot
    plt.figure(figsize=figsize)
    plt.plot(pd.Series([i for i in range(1, len(signal))]), 
             signal[1:], color=color, alpha=0.7)
    plt.ylabel(y_units)
    plt.xlabel(x_units)
    plt.title(title_line)
    plt.show();
    
    
# Using Numpy instead of Pandas to create the 1-dimensional arrays
def numpy_train_test_split(data_folder, train_set, test_set):
    """
    create target classes for training and test data using numpy
    """
    import numpy as np
    
    train = np.loadtxt(data_folder+train_set, skiprows=1, delimiter=',')
    x_train = train[:, 1:]
    y_train = train[:, 0, np.newaxis] - 1.
    
    test = np.loadtxt(data_folder+test_set, skiprows=1, delimiter=',')
    x_test = test[:, 1:]
    y_test = test[:, 0, np.newaxis] - 1.
    
    train,test
    
    return x_train, y_train, x_test, y_test

def zero_scaler(x_train, x_test):
    """
    Scales each observation of an array to zero mean and unit variance.
    Takes array for train and test data separately.
    """
    import numpy as np
        
    x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 
           np.std(x_train, axis=1).reshape(-1,1))
    
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
              np.std(x_test, axis=1).reshape(-1,1))
 
    return x_train, x_test

def time_filter(x_train, x_test, step_size=None, axis=2):
    """
    Adds an input corresponding to the running average over a set number
    of time steps. This helps the neural network to ignore high frequency 
    noise by passing in a uniform 1-D filter and stacking the arrays. 
    
    **ARGS
    step_size: integer, # timesteps for 1D filter. defaults to 200
    axis: which axis to stack the arrays
    """
    import numpy as np
    from scipy.ndimage.filters import uniform_filter1d
    
    if step_size is None:
        step_size=200
    
    train_filter = uniform_filter1d(x_train, axis=1, size=step_size)
    test_filter = uniform_filter1d(x_test, axis=1, size=step_size)
    
    x_train = np.stack([x_train, train_filter], axis=2)
    x_test = np.stack([x_test, test_filter], axis=2)
#     x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, 
#                                                  size=time_steps)], axis=2)
#     x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, 
#                                                size=time_steps)], axis=2)
    
    return x_train, x_test


def batch_maker(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples rotating randomly
    
    generator: A generator or an instance of `keras.utils.Sequence`
        
    The output of the generator must be either
    - a tuple `(inputs, targets)`
    - a tuple `(inputs, targets, sample_weights)`.

    This tuple (a single output of the generator) makes a single
    batch. Therefore, all arrays in this tuple must have the same
    length (equal to the size of this batch). Different batches may have 
    different sizes. 

    For example, the last batch of the epoch
    is commonly smaller than the others, if the size of the dataset
    is not divisible by the batch size.
    The generator is expected to loop over its data
    indefinitely. An epoch finishes when `steps_per_epoch`
    batches have been seen by the model.
    
    """
    import numpy
    import random

    half_batch = batch_size // 2
    
    # Returns a new array of given shape and type, without initializing entries.
    # x_train.shape = (5087, 3197, 2)
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    
    #y_train.shape = (5087, 1)
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    
    pos_idx = np.where(y_train[:,0] == 1.)[0]
    neg_idx = np.where(y_train[:,0] == 0.)[0]

    # rotating each of the samples randomly
    while True:
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
    
        x_batch[:half_batch] = x_train[pos_idx[:half_batch]]
        x_batch[half_batch:] = x_train[neg_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[pos_idx[:half_batch]]
        y_batch[half_batch:] = y_train[neg_idx[half_batch:batch_size]]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch
        
        
        
# def scikit_keras(build_fn=None, compiler=None, params=None, batch_size=32):
#     """
#     Builds, compiles and fits a keras model
#     Takes in dictionaries of parameters for both compiler and
#     fit_generator.
    
#     *ARGS
#     build_fn: build function for creating model, can also pass in a model
#     compiler : dict of paramaters for model.compile()
#     params : dict of parameters for model.fit_generator
#     note: batch
    
    
#     """
#     # set default parameters if not made explicit
    
#     # BUILD vars
#     if build_fn:
#         model=build_fn
#     else:
#         model = keras_1D(model=Sequential(), kernel_size=11, activation='relu', 
#                            input_shape=x_train.shape[1:], strides=4)

#     # COMPILE vars
#     if compiler:   
#         optimizer=compiler['optimizer']
#         learning_rate=compiler['learning_rate'] 
#         loss=compiler['loss']
#         metrics=compiler['metrics']
     
#     else:
#         optimizer=Adam
#         learning_rate=1e-5
#         loss='binary_crossentropy'
#         metrics=['accuracy']
        
        
#     ##### COMPILE AND FIT #####
#     model.compile(optimizer=optimizer(learning_rate), loss=loss, 
#                   metrics=metrics)
    
#     # HISTORY vars
# #     if generator is None:
# #         generator = batch_maker(x_train, y_train, batch_size)
    
#     if params:
#         validation_data = params['validation_data']
#         verbose = params['verbose']
#         epochs = params['epochs']
#         steps_per_epoch = params['steps_per_epoch']
#     else:
#         validation_data = (x_test, y_test)
#         verbose=0
#         epochs=5
#         steps_per_epoch=x_train.shape[1]//32
    
#     history = model.fit_generator(batch_maker(x_train, y_train, batch_size), 
#                                   validation_data=validation_data, 
#                                   verbose=verbose, epochs=epochs, 
#                                   steps_per_epoch=steps_per_epoch)
    
#     return model, history


# Build these values into a function for efficiency in next model iterations:

def get_preds(x_test,y_test,model=None,**kwargs):
    #y_true = (y_test[:, 0] + 0.5).astype("int") # flatten and make integer
    #y_hat = model.predict(x_test)[:,0] 
    
    y_true = y_test.flatten()
    y_pred = model.predict_classes(x_test).flatten() # class predictions 
    
    
    yhat_val = pd.Series(y_pred).value_counts(normalize=False)
    yhat_pct = pd.Series(y_pred).value_counts(normalize=True)*100

    print(f"y_hat_vals:\n {yhat_val}")
    print("\n")
    print(f"y_pred:\n {yhat_pct}")
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    print('\nAccuracy Score:', acc)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print("\nConfusion Matrix")
    display(cm)
    return y_true,y_pred
    
    
def plot_keras_history(history,figsize=(10,4),subplot_kws={}):
    if hasattr(history,'history'):
        history=history.history
    figsize=(10,4)
    subplot_kws={}

    acc_keys = list(filter(lambda x: 'acc' in x,history.keys()))
    loss_keys = list(filter(lambda x: 'loss' in x,history.keys()))

    fig,axes=plt.subplots(ncols=2,figsize=figsize,**subplot_kws)
    axes = axes.flatten()

    y_labels= ['Accuracy','Loss']
    for a, metric in enumerate([acc_keys,loss_keys]):
        for i in range(len(metric)):
            ax = pd.Series(history[metric[i]],
                        name=metric[i]).plot(ax=axes[a],label=metric[i])
    [ax.legend() for ax in axes]
    [ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True)) for ax in axes]
    [ax.set(xlabel='Epochs') for ax in axes]
    plt.suptitle('Model Training Results',y=1.01)
    plt.tight_layout()
    plt.show()
    
    
# PLOT Confusion Matrices

def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    
    import itertools
    # Check if normalize is set to True
    # If so, normalize the raw confusion matrix before visualizing
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    
    fig, ax = plt.subplots(figsize=(10,10))
    #mask = np.zeros_like(cm, dtype=np.bool)
    #idx = np.triu_indices_from(mask)
    
    #mask[idx] = True

    plt.imshow(cm, cmap=cmap, aspect='equal')
    
    # Add title and axis labels 
    plt.title('Confusion Matrix') 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #ax.set_ylim(len(cm), -.5,.5)
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    thresh = cm.max() / 2.
    # iterate thru matrix and append labels  
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='darkgray' if cm[i, j] > thresh else 'black',
                size=14, weight='bold')
    
    # Add a legend
    plt.colorbar()
    plt.show() 
    
    
def roc_plots(y_test, y_hat):
    from sklearn import metrics
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
    y_true = (y_test[:, 0] + 0.5).astype("int")   
    fpr, tpr, thresholds = roc_curve(y_true, y_hat) 
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)

    # Threshold Cutoff for predictions
    crossover_index = np.min(np.where(1.-fpr <= tpr))
    crossover_cutoff = thresholds[crossover_index]
    crossover_specificity = 1.-fpr[crossover_index]
    #print("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))
    
    plt.plot(thresholds, 1.-fpr)
    plt.plot(thresholds, tpr)
    plt.title("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))

    plt.show()


    plt.plot(fpr, tpr)
    plt.title("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
    plt.show()
    
    score = roc_auc_score(y_true,y_hat)
    print("ROC_AUC SCORE:",score)
    #print("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
    
    
def evaluate_model(x_test, y_test, history=None):
    
    # make predictons using test set
    y_true = (y_test[:, 0] + 0.5).astype("int") # flatten and make integer
    y_hat = model.predict(x_test)[:,0] 
    y_pred = model.predict_classes(x_test).flatten() # class predictions 
    
    
    #Plot Model Training Results (PLOT KERAS HISTORY)
    from sklearn import metrics
    if y_true.ndim>1:
        y_true = y_true.argmax(axis=1)
    if y_pred.ndim>1:
        y_pred = y_pred.argmax(axis=1)   
    try:    
        if history is not None:
            plot_keras_history(history)
    except:
        pass
    
    # Print CLASSIFICATION REPORT
    num_dashes=20
    print('\n')
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)
#     try:
#         print(metrics.classification_report(y_true,y_pred))
         #fig = plot_confusion_matrix((y_true,y_pred))
#     except Exception as e:
#         print(f"[!] Error during model evaluation:\n\t{e}")

    from sklearn import metrics
    report = metrics.classification_report(y_true,y_pred)
    print(report)
    
    # Adding additional metrics not in sklearn's report   
    from sklearn.metrics import jaccard_score
    jaccard = jaccard_score(y_test, y_hat_test)
    print('Jaccard Similarity Score:',jaccard)
    
    
    # CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # Plot normalized confusion matrix
    fig = plot_confusion_matrix(cm, classes=['No Planet', 'Planet'], 
                                normalize=False,                               
                                title='Normalized confusion matrix')
    plt.show()

    
    # ROC Area Under Curve
    roc_plots(y_test, y_hat_test)
    