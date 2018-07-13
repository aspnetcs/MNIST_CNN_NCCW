function netResult = buildTrainTestCNN(convLayers,numFilters,batchNormalisation,maxPooling,options,train,test4D,testCell)
%buildTrainTestCNN Takes network parameters, training and testing data and
% returns test results
%   Takes as input: convLayers - number of convolutional layers required
%                   numFilters - the number of filters required in each
%                   convolutional layer
%                   batchNormalisation - inclusion of batch normalisation
%                   layers (boolean)
%                   maxPooling - inclusion of max-pooling layers (boolean)
%                   options - network training options
%                   train - training data
%                   test4D - a 4D matrix of test images
%                   testCell - a cell array of test images and ground truth
%                   classifications

% Build the model
layers = buildLayers(convLayers,numFilters,batchNormalisation,maxPooling);
% Train it (with timing)
tic;
[net, netInfo] = trainNetwork(train,layers,options);
trainingTime = toc;
% Get training and validation results
trainAccuracy = netInfo.TrainingAccuracy(end);
valAccuracy = netInfo.ValidationAccuracy(netInfo.ValidationAccuracy > 0);
valAccuracy = valAccuracy(end);
% Test it
yhat = classify(net,test4D);
testAccuracy = sum(yhat == testCell.y)/numel(testCell.y)*100;
netResult = {convLayers,numFilters,batchNormalisation,maxPooling,...
    trainAccuracy,valAccuracy,testAccuracy,trainingTime};
% Create the name for the net and save it to file
if batchNormalisation == true
    BN = 'T';
else
    BN = 'F';
end
if maxPooling == true
    MP = 'T';
else
    MP = 'F';
end
netName = strcat('CNN_L',num2str(convLayers),'_F',num2str(numFilters),'_BN',BN,'_MP',MP);
save(netName, 'net');
end

