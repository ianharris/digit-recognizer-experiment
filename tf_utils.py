import tensorflow as tf


"""
Function to create weight variables in the neural network
"""
def initialise_weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

"""
Function to create bias variables in the neural network
"""
def initialise_bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

"""
Function to perform a convolution
"""
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

"""
Function to perform max pooling
"""
def maxpool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

"""
Function to apply a fully connected layer
"""
def fully_connected_layer(input_dims, output_dims, x):
    
    # create the weight and bias variables
    W = initialise_weight_variable([input_dims, output_dims])
    b = initialise_bias_variable([output_dims])

    return tf.matmul(x, W) + b

"""
Function to apply a fully connected layer with ReLU and dropout
"""
def fully_connected_layer_with_relu_and_dropout(input_dims, output_dims, x):
    
    # create the fully connected layer
    l = fully_connected_layer(input_dims, output_dims, x)

    r = tf.nn.relu(l)
    
    keep_probability = tf.placeholder(tf.float32)
    d = tf.nn.dropout(r, keep_probability)

    return d, keep_probability


"""
Function to create a convolutional layer with relu
"""
def convolutional_layer_with_relu(filter_height, filter_width, input_dims, output_dims, x):
    
    # create weight and bias variables
    W = initialise_weight_variable([filter_height, filter_width, input_dims, output_dims])
    b = initialise_bias_variable([output_dims])

    # perform the convolution
    c = conv2d(x, W) + b

    # perform relu activation
    r = tf.nn.relu(c)

    # perform max_pooling
    m = maxpool_2x2(r)

    return m
    
"""
Function to create a convolutional layer with relu
"""
def convolutional_layer_with_leaky_relu(filter_height, filter_width, input_dims, output_dims, x, alpha):
    
    # create weight and bias variables
    W = initialise_weight_variable([filter_height, filter_width, input_dims, output_dims])
    b = initialise_bias_variable([output_dims])

    # perform the convolution
    c = conv2d(x, W) + b

    # perform leaky relu activation
    r = tf.nn.leaky_relu(c, alpha=alpha)

    # perform max_pooling
    m = maxpool_2x2(r)

    return m
    
