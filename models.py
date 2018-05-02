from tf_utils import *

def get_train_and_accuracy(y,yconv):
    # create the loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yconv))
    # create the train step
    train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # check if the prediction was right
    correct_prediction= tf.equal(tf.argmax(yconv, 1), tf.argmax(y, 1))
    # measure the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train, accuracy

def get_models(x, y):
 
    # build the 'zeroth' model - this is the Tensorflow Tutorial CNN architecture
    m0desc = """
     * one convolutional layer with 32 filters and ReLU activation
     * one convolutional layer with 64 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c0a = convolutional_layer_with_relu(5, 5, 1, 32, x)
    c0b = convolutional_layer_with_relu(5, 5, 32, 64, c0a)
    c0b_flat = tf.reshape(c0b, [-1, 7 * 7 * 64])
    fc0a, kp0a = fully_connected_layer_with_relu_and_dropout(7 * 7 * 64, 1024, c0b_flat)
    fco0 = fully_connected_layer(1024, 10, fc0a)
    (t0, acc0) = get_train_and_accuracy(y, fco0)
    m0 = (m0desc, t0, acc0, kp0a)

    # build the first model
    m1desc = """
     * one convolutional layer with 32 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c1a = convolutional_layer_with_relu(5, 5, 1, 32, x)
    c1a_flat = tf.reshape(c1a, [-1, 14 * 14 * 32])
    fc1a, kp1a = fully_connected_layer_with_relu_and_dropout(14 * 14 * 32, 1024, c1a_flat)
    fco1 = fully_connected_layer(1024, 10, fc1a)
    (t1, acc1) = get_train_and_accuracy(y, fco1)
    m1 = (m1desc, t1, acc1, kp1a)

    # build the second model
    m2desc = """
     * one convolutional layer with 64 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c2a = convolutional_layer_with_relu(5, 5, 1, 64, x)
    c2a_flat = tf.reshape(c2a, [-1, 14 * 14 * 64])
    fc2a, kp2a = fully_connected_layer_with_relu_and_dropout(14 * 14 * 64, 1024, c2a_flat)
    fco2 = fully_connected_layer(1024, 10, fc2a)
    (t2, acc2) = get_train_and_accuracy(y, fco2)
    m2 = (m2desc, t2, acc2, kp2a)

    # build the third model - this is the Tensorflow Tutorial CNN architecture except the convolutional layers use leaky ReLU activation
    m3desc = """
     * one convolutional layer with 32 filters and leaky ReLU activation
     * one convolutional layer with 64 filters and leaky ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c3a = convolutional_layer_with_leaky_relu(5, 5, 1, 32, x, 0.2)
    c3b = convolutional_layer_with_leaky_relu(5, 5, 32, 64, c3a, 0.2)
    c3b_flat = tf.reshape(c3b, [-1, 7 * 7 * 64])
    fc3a, kp3a = fully_connected_layer_with_relu_and_dropout(7 * 7 * 64, 1024, c3b_flat)
    fco3 = fully_connected_layer(1024, 10, fc3a)
    (t3, acc3) = get_train_and_accuracy(y, fco3)
    m3 = (m3desc, t3, acc3, kp3a)

    # build the fourth model
    m4desc = """
     * one convolutional layer with 32 filters and leaky ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c4a = convolutional_layer_with_leaky_relu(5, 5, 1, 32, x, 0.2)
    c4a_flat = tf.reshape(c4a, [-1, 14 * 14 * 32])
    fc4a, kp4a = fully_connected_layer_with_relu_and_dropout(14 * 14 * 32, 1024, c4a_flat)
    fco4 = fully_connected_layer(1024, 10, fc4a)
    (t4, acc4) = get_train_and_accuracy(y, fco4)
    m4 = (m4desc, t4, acc4, kp4a)
    
    # build the fifth model
    m5desc = """
     * one convolutional layer with 64 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c5a = convolutional_layer_with_leaky_relu(5, 5, 1, 64, x, 0.2)
    c5a_flat = tf.reshape(c5a, [-1, 14 * 14 * 64])
    fc5a, kp5a = fully_connected_layer_with_relu_and_dropout(14 * 14 * 64, 1024, c5a_flat)
    fco5 = fully_connected_layer(1024, 10, fc5a)
    (t5, acc5) = get_train_and_accuracy(y, fco5)
    m5 = (m5desc, t5, acc5, kp5a)

    # build the sixth model
    m6desc = """
     * one convolutional layer with 1 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c6a = convolutional_layer_with_relu(5, 5, 1, 1, x)
    c6a_flat = tf.reshape(c6a, [-1, 14 * 14 * 1])
    fc6a, kp6a = fully_connected_layer_with_relu_and_dropout(14 * 14 * 1, 1024, c6a_flat)
    fco6 = fully_connected_layer(1024, 10, fc6a)
    (t6, acc6) = get_train_and_accuracy(y, fco6)
    m6 = (m6desc, t6, acc6, kp6a)

    # build the seventh model
    m7desc = """
     Multiclass Perceptron
     * no hidden layers input fully connected to output
    """
    x7_reshape = tf.reshape(x, [-1, 784])
    fco7 = fully_connected_layer(784, 10, x7_reshape)
    (t7, acc7) = get_train_and_accuracy(y, fco7)
    m7 = (m7desc, t7, acc7, None)

    # build the 8th model
    m8desc = """
     * one convolutional layer with 2 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c8a = convolutional_layer_with_relu(5, 5, 1, 2, x)
    c8a_flat = tf.reshape(c8a, [-1, 14 * 14 * 2])
    fc8a, kpa8 = fully_connected_layer_with_relu_and_dropout(14 * 14 * 2, 1024, c8a_flat)
    fco8 = fully_connected_layer(1024, 10, fc8a)
    (t8, acc8) = get_train_and_accuracy(y, fco8)
    m8 = (m8desc, t8, acc8, kpa8)

    # build the 9th model
    m9desc = """
     * one convolutional layer with 4 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c9a = convolutional_layer_with_relu(5, 5, 1, 4, x)
    c9a_flat = tf.reshape(c9a, [-1, 14 * 14 * 4])
    fc9a, kpa9 = fully_connected_layer_with_relu_and_dropout(14 * 14 * 4, 1024, c9a_flat)
    fco9 = fully_connected_layer(1024, 10, fc9a)
    (t9, acc9) = get_train_and_accuracy(y, fco9)
    m9 = (m9desc, t9, acc9, kpa9)

    # build the 10th model
    m10desc = """
     * one convolutional layer with 8 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c10a = convolutional_layer_with_relu(5, 5, 1, 8, x)
    c10a_flat = tf.reshape(c10a, [-1, 14 * 14 * 8])
    fc10a, kpa10 = fully_connected_layer_with_relu_and_dropout(14 * 14 * 8, 1024, c10a_flat)
    fco10 = fully_connected_layer(1024, 10, fc10a)
    (t10, acc10) = get_train_and_accuracy(y, fco10)
    m10 = (m10desc, t10, acc10, kpa10)

    # build the 11th model
    m11desc = """
     * one convolutional layer with 16 filters and ReLU activation
     * one fully connected layer with 1024 neurons, ReLU and dropout
     * an output layer
    """
    c11a = convolutional_layer_with_relu(5, 5, 1, 16, x)
    c11a_flat = tf.reshape(c11a, [-1, 14 * 14 * 16])
    fc11a, kpa11= fully_connected_layer_with_relu_and_dropout(14 * 14 * 16, 1024, c11a_flat)
    fco11 = fully_connected_layer(1024, 10, fc11a)
    (t11, acc11) = get_train_and_accuracy(y, fco11)
    m11 = (m11desc, t11, acc11, kpa11)

    return [ m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11 ]

