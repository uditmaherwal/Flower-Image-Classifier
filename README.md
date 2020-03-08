# Flower-Image-Classifier

In this project, I trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. I used the dataset of 102 flower categories.

The project is broken down into multiple steps:
1. Load and preprocess the image dataset
2. Train the image classifier on your dataset
3. Use the trained classifier to predict image content
I'll lead you through each part which I've implemented in Python.

• Challenge: Udacity Data Scientist Nanodegree project for deep learning module titled as 'Image Classifier with Deep Learning' attempts to train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice we had to train this classifier, then export it for use in our application. We had used a dataset (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

• Solution: Used torchvision to load the data. The dataset is split into three parts, training, validation, and testing. For the training, applied transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. Also need to load in a mapping from category label to category name. Wrote inference for classification after training and testing the model. Then processed a PIL image for use in a PyTorch model.

• Result: Using the following software and Python libraries: Torch, PIL, Matplotlib.pyplot, Numpy, Seaborn, Torchvision. Thus, achieved an accuracy of 80% on test dataset as a result of above approaches. Performed a sanity check since it's always good to check that there aren't obvious bugs despite achieving a good test accuracy. Plotted the probabilities for the top 5 classes as a bar graph, along with the input image.

Software and Libraries
This project uses the following software and Python libraries:
NumPy, pandas, Matplotlib (for data visualization), Seaborn.

Code File
Open file jupyter notebook imageclassifierproject.ipynb
