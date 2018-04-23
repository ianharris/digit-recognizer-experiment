# base imports
import sys
import getopt

# numerical packages / tensorflow
import random
import pandas as pd
import numpy as np
import tensorflow as tf

# user defined code imports
from data import Data
from models import get_models
from tf_utils import *

verbose = False


def usage():
    print('Usage')
    print('python %s                : runs the set of models' % sys.argv[0])
    print('python %s -v|--verbose   : runs the set of models with verbose output' % sys.argv[0])
    print('python %s -h             : prints this help and exits' % sys.argv[0])

def run_model(train, accuracy, kp, x, y, vbatch, dtrain):
 
    global verbose

    # create a validation feed dictionary for this model run in the ensemble
    v_feed_dict = { x: vbatch[0], y: vbatch[1] }

    # iterate over batches for training
    for i in range(5000):
        
        # get a batch
        batch = dtrain.get_batch()
        
        # create the training feed dict
        tr_feed_dict = { x: batch[0], y: batch[1] }

        # report accuracy if i modulo 1000 is zero - i.e. every 1000th step - includes first step
        if i % 1000 == 0 and verbose:
            
            # set a value to feed kp if it has been set
            if kp is not None:
                tr_feed_dict[kp] = 1.0

            train_acc = accuracy.eval(feed_dict=tr_feed_dict)
            print('step %d, training accuracy %g' % (i, train_acc))

            # set a value to feed kp if it has been set
            if kp is not None:
                v_feed_dict[kp] = 1.0

            valid_acc = accuracy.eval(feed_dict=v_feed_dict)
            print('Validation accuracy %g' % valid_acc)

        # set a value to feed kp if it has been set
        if kp is not None:
            tr_feed_dict[kp] = 0.5

        # run a training step
        train.run(feed_dict=tr_feed_dict)

    # get the final accuracy
    # set a value to feed kp if it has been set
    if kp is not None:
        v_feed_dict[kp] = 1.0

    valid_acc = accuracy.eval(feed_dict=v_feed_dict)
    if verbose:
        print('\nFinal Validation accuracy: %g\n' % valid_acc)

    return valid_acc

    

def run_ensemble(sess, m, x, y, vbatch, dtrain):
    
    # print the model description to screen
    print('======================')
    print(m[0])
    
    # get the accuracy
    accuracy = m[2]
    # get the trainer
    train = m[1]
    # get the keep probability
    kp = m[3]

    
    acc = []
    ensemble_count = 3

    for i in range(ensemble_count):
        
        # initialise global variables in the graph
        sess.run(tf.global_variables_initializer())

        acc.append(run_model(train, accuracy, kp, x, y, vbatch, dtrain))

    print(acc)
    print('\n\nEnsemble average validation accuracy: %g\n\n' % (sum(acc)/len(acc)))
    print('======================')

if __name__ == '__main__':

    # get command line options
    optlist, args = getopt.getopt(sys.argv[1:], 'vh', ['verbose', 'help'])

    for (k, v) in optlist:
        if k in ('-v', '--verbose'):
            verbose = True
        elif k in ('-h', '--help'):
            usage()
            sys.exit(0)

    # create an instance of the Data class
    dtrain = Data('datasets/train-exploration.csv')
    dvalid = Data('datasets/validation-exploration.csv')
    
    # create placeholders for x and y
    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    # build the neural network
    models = get_models(x, y)

    with tf.Session() as sess:
        
        # get a validation batch
        vbatch = (dvalid.features, dvalid.labels) # get_batch(4207)

        # iterate over all models
        for m in models:
 
            run_ensemble(sess, m, x, y, vbatch, dtrain)
        
