# MNIST_CNN_NCCW
Application of a CNN to MNIST using MATLAB. One half of a coursework comparing and contrasting a CNN with an SVM.

In the experiment I performed grid searches, using 10% of the dataset, first to explore the impact of including/
excluding batch normalisation and max-pooling layers, and then to look into the effects of changing the depth and
number of filters in the network. I then selected the best performing parameter combinations and trained a final 
network on the full dataset to compare to my partners SVM model.


## CNN Scripts(to be run in order):
TS_NC_CW_main.m - The main CNN script. Takes the original MNIST data and converts its to a matlab useable format.\\
		  Takes a random 10% sample from each dataset. Puts this and the full dataset in to an appropriate
		  format for the CNN model to use.
		  Builds and tests the final 4 conv layer, 64 filter model.\\
TS_NC_CW_Exp.m  - Trains and tests all CNN models required in the architecture and layer depth/size grid searches,
		  using the 10% sample datasets.\\
TS_NC_CW_Plotting.m - Creates graphs (not used in final report) and heatmaps of experiment results.
		  Extracts filter visualisations and incorrect classifications from the final CNN model.

## CNN Functions:
buildTrainTestCNN.m - Function takes in parameter settings and data, constructs a CNN, tests it, saves the model 
		  to file and returns the results. Required for all three TS_NC_CW scripts.\\
buildLayers.m   - Function takes in parameter settings and returns a layer vector for a CNN. Required for 
		  buildTrainTestCNN.\\
convertMNIST.m  - Function to extract the MNIST data and convert to .mat format. Used courtesy of 
		  https://github.com/sunsided/mnist-matlab