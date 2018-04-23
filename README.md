# Overview

I recently started looking at Kaggle's [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer) competition. The competition is a knowledge competition to create a model that can correctly identify hand written digits. The data set for the challenge is the well-known [MNIST data set](http://yann.lecun.com/exdb/mnist/). I initially created a Convlutional Neural Network (CNN) based on the CNN presented in Tensorflow's [Deep MNIST Tutorial](https://www.tensorflow.org/versions/r1.2/get_started/mnist/pros); the code for the Tensorflow Tutorial can be viewed [here](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/mnist/mnist_deep.py). 

In my implementation, I mimicked the CNN structure from Tensorflow example. Specifically, the following items are taken from Tensorflow's Tutorial implementation:

* the use of two convolutional layers
* the use of 32 filters in the first convolutional layer and 64 filters in the second convolutional layer
* the use of 2x2 max pooling in both convolutional layers
* the use of ReLU activation in both convolutional layers
* the use of a fully connected hidden layer with ReLU and Dropout (with 50% probability)
* the use of Softmax with a Cross Entropy loss
* the use of the Adam optimizer

This implementation does quite well (predicting with ~ 98.5% accuracy) at Kaggle's Digit Recognizer challenge; which is unsurprising given the Tensorflow team built the model. But having leveraged it, I'm left with the question why does it do so well? What are the key components of this model that make it so effective. Is it the number of convolutional layers? The number of filters in each layer? The use of the ReLU activation function? I want to experiment a little with the model to see if I can answer this question.

This repo contains code that allows multiple models to be run to assess their accuracy.


# Training and Validation Data Set 

In order to explore the difference in CNN architectures on the results we need to have a proper training and validation set. The data provided as part of the Kaggle competition is certainly a good start but has one draw back; we don't have the labels for the validation set. 

Note: *Kaggle refer to this set as the test set. The term validation is more appropriate for our use as we want it to use it to validate various model choices rather than test a selected model.*

To correctly compare the results of various architectures a set of labelled validation data is a must; we could evaluate the results by predicting on the Kaggle test data set and then submit to Kaggle but that would limit us to 10 validations a day.

To overcome this issue we will use the Kaggle training set and split it up into a new training and validation set. The Kaggle training set has 42,000 labelled images. We will take 10% of this as a validation set and keep the other 90% as a training set. When we divide up the set we need to be sure we maintain uniformity - we don't want the validation set to consist only of the digit 9, for example.

The Jupyter Notebook 'DataExtraction.ipynb' contains the code for extracting the data. It assumes that the 'train.csv' file from Kaggle's Digit Recognizer competition has been downloaded to the 'datasets' subdirectory. The notebook complete with output can be viewed at [docs/DataExtraction.md](docs/DataExtraction.md).

# Areas of Investigation

There are many parts of the CNN architecture that we could vary so we need to limit ourselves to certain areas. Specifically, we will __not__ investigate the following parts of the architecture (at least at this point):

* the use of Softmax with a Cross Entropy loss
* the use of the Adam optimizer

# Training Methodology and Length

For the purposes of the experiment we will train using batches of 50 images. We will want to choose a number of batch iterations that is both high enough to train to a high level of accuracy but also low enough to ensure training doesn't take excessively long on a laptop.

A high level analysis of number of iterations vs accuracy is presented in the Jupyter Notebook AccuracyPlot.ipynb. The notebook completed with output can be viewed at [docs/AccuracyPlot.md](docs/AccuracyPlot.md)

# Pre-requisites

Before running the models the Kaggle Digit Recognizer training dataset (train.csv) should be downloaded and saved to the datasets directory. The Notebook DataExtraction.ipynb should then be run to produce the new training and validation datasets.

# Running the models

The models can be run by typing:

```python
python mnist.py
```

A full run of all eight models with ensembles of three runs can take several hours on a laptop.

# Results

The table below outlines the results from an initial run of models. The models include a variety of CNN architectures as well as one perceptron architecture. 

The accuracy of each model was calculated using an ensemble of three model runs. By using ensembles we get an extra check that the models don't suffer from running for a relatively short number of iterations. Specifically, the effect of randomness introduced by initialisation of weight and bias values in the network as well as the use of dropout. At the early iterations of training this randomness could have a significant effect on the models calculated weights and bias's. Of course, it is expected that after a sufficient number of iterations, the effect of initial conditions or dropout selection will diminish in favour of the optimizer guiding the model towards the loss functions minima.

As can be seen from the table the accuracies of each model run varied by less than a percentage point for a given model, indicating that the models did run for a sufficient number of iterations to overcome any effects of randomness.

Somewhat surprisingly, almost all of the CNNs achieved comparable validation accuracies; in particular, validation accuracies between ~0.96 and ~0.98. The exception being model 6; the CNN that contained a single convolutional layer with only one filter. For model 6 the achieved validation accuracy was ~0.86 which, while still high, is significantly lower than the other CNN models at that level of accuracy. Model 6 is also comparable in accuracy to model 7, which is a very simple multiclass perceptron where each input is connected to an output; a model as simple as model 7 wouldn't be expected to perform very well.

| Model Number | Model Description | Ensemble Results | Ensemble Average |
|:------------:|:------------------|:-----------------|:-----------------|
|0             |<ul><li>one convolutional layer with 32 filters and ReLU activation</li><li>one convolutional layer with 64 filters and ReLU activation</li><li>one fully connected layer with 1024 neurons, ReLU and dropout</li><li>an output layer</li></ul>| [0.97360915, 0.97289586, 0.97360915] | 0.973371 |
|1             |<ul><li>one convolutional layer with 32 filters and ReLU activation</li><li>one fully connected layer with 1024 neurons, ReLU and dropout</li><li>an output layer</li></ul> | [0.96647644, 0.96837848, 0.97146934] | 0.968775 |
|2             |<ul><li>one convolutional layer with 64 filters and ReLU activation</li><li>one fully connected layer with 1024 neurons, ReLU and dropout</li><li>an output layer</li></ul> | [0.9726581, 0.96909177, 0.96885401] | 0.970201 |
|3             |<ul><li>one convolutional layer with 32 filters and leaky ReLU activation</li><li>one convolutional layer with 64 filters and leaky ReLU activation</li><li>one fully connected layer with 1024 neurons, ReLU and dropout</li><li>an output layer</li></ul> | [0.97170711, 0.97242033, 0.97432238] | 0.972817 | 
|4             |<ul><li>one convolutional layer with 32 filters and leaky ReLU activation</li><li>one fully connected layer with 1024 neurons, ReLU and dropout</li><li>an output layer</li></ul> | [0.96885401, 0.96861625, 0.96600097] | 0.967824 |
|5             |<ul><li>one convolutional layer with 64 filters and ReLU activation</li><li>one fully connected layer with 1024 neurons, ReLU and dropout</li><li>an output layer</li></ul> | [0.96790302, 0.97194487, 0.97432238] | 0.97139 |
|6             |<ul><li>one convolutional layer with 1 filters and ReLU activation</li><li>one fully connected layer with 1024 neurons, ReLU and dropout</li><li>an output layer</li></ul> | [0.85520685, 0.8673324, 0.85354257] | 0.858694 |
|7             |<ul><li>no hidden layers input fully connected to output</li></ul> | [0.85068947, 0.84854966, 0.86162627] | 0.853622 |

# Conclusion

In this initial set of model runs we saw that reducing the number of convolutional layers in a CNN from two to one had no significant effect on model accuracy as long as the number of filters in the convoluational layer remained high; in our case 32 or 64. When the number of filters in a CNN with a single convolutional layer was reduced to one, the model performed as poorly as a multiclass perceptron.

The effect of using ReLU vs leaky ReLU activation functions was also negligible for CNNs with multiple fitlers in their layers.

With the current results in mind, the next set of investigations should focus on:

* CNNs with a single convolutional layer with low levels of filters e.g. [ 1, 2, 4, 8, 16 ]
* Varying the dropout in the network e.g. [ 0.3, 0.5, 0.7 ]



