function layers = buildLayers(convLayers,numFilters,batchNormalisation,maxPooling)
%buildLayers Constructs layer vector according to input parameters
%   buildLayers is used to define the atchitecture of a Convolutional
%   Neural Net. The standard structure is:
%                   Input layer
%                   Mid layer group:
%                       Convolutional layer
%                       Batch normalisation layer
%                       ReLU activation layer
%                       Max pooling layer (not present in final group)
%                   Fully connected layer
%                   Softmax activation layer
%                   Classification layer
%
%   The number of mid layer groups is determined by the convLayers
%   parameter. The number of filters in each convolutional layer is
%   determined by the numFilters parameter.
%   Batch normalisation and max pooling layers can be switched on or off
%   according to the boolean batchNormalisation and maxPooling parameters.

% Determine whether or not to include batch normalisation layers
if batchNormalisation == true
    batchLayer = batchNormalizationLayer;
else
    batchLayer = [];
end
% Determine whether or not to include max-pooling layers
if maxPooling == true
    maxPoolLayer = maxPooling2dLayer(3,'Stride',2);
else
    maxPoolLayer = [];
end

midLayers = [
    convolution2dLayer(3,numFilters,'Padding',1)
    batchLayer
    reluLayer
    maxPoolLayer];

altMidLayers = [
    convolution2dLayer(3,numFilters,'Padding',1)
    batchLayer
    reluLayer];

totalMidLayers = [];
% If there are 3 or fewer convolutional layers, include a maxpool layer
% after if each one (if maxPooling = true)
% Else, only include maxpooling layers after the final three.
% This is required because maxpooling reduces the size of the layer output.
% Including more than three maxpooling layers reduces the output size too
% far (to the point where it is smaller than the filter size).
if convLayers < 4
    for i = 1:(convLayers)
        totalMidLayers = [totalMidLayers;midLayers];
    end
else
    for i = 1:3
        totalMidLayers = [totalMidLayers;midLayers];
    end
    for i = 4:convLayers
        totalMidLayers = [altMidLayers;totalMidLayers];
    end
end

layers = [
    imageInputLayer([28 28 1])
    
    totalMidLayers
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
    
end

