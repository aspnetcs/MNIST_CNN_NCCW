%% Load the pre-arranged dataset

clc; clear; close all; 
load('miniMNIST.mat')
%% Create Results Table

headers = {'convLayers','numFilters','batchNormalisation','maxPooling',...
    'trainAccuracy','valAccuracy','testAccuracy','TrainingTime'};
results = cell(24,8);
results = cell2table(results);
results.Properties.VariableNames = headers;

% Training Options
options = trainingOptions('sgdm',...
    'MaxEpochs',5, ...
    'ValidationData',miniVal,...
    'ValidationFrequency',30,...
    'ValidationPatience',3,...
    'Verbose',false,...
    'Plots','training-progress');

%% Experiment with presence of batch normalisation and max pooling
convLayers = 3;
numFilters = 16;
i = 1;

for b = 0:1 % For batch normalisation [true false]
    batchNormalisation = b;
    for m = 0:1 % For max-pooling [true false]
        maxPooling = m;
        % Build, train and test a model
        netResults = buildTrainTestCNN(convLayers,numFilters,...
            batchNormalisation,maxPooling,options,miniTrain,miniTest4D,miniTest);
        % Record the results
        results{i,:} = netResults;
        i = i+1;
    end
end

%% Experiment with number of layers and number of filters

batchNormalisation = true;
maxPooling = true;

for layer = 1:5 % For each network depth
    convLayers = layer;
    for filter = [16 32 64 128] % For each conv layer size
        numFilters = filter;
        % Build, train and test a model
        netResults = buildTrainTestCNN(convLayers,numFilters,...
            batchNormalisation,maxPooling,options,miniTrain,miniTest4D,miniTest);
        % Record the results
        results{i,:} = netResults;
        i = i+1;
    end
end

%% Save the resuts
save('FullResults.mat','results');

