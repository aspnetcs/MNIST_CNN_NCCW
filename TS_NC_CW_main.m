%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Neural Computing Coursework
%  Toby Staines
%  08/03/18
%  Application of a Convolutional Neural Network (CNN) to the MNIST dataset 
%  (for comparison with a Support Vector Machine)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load the MNIST data

% convertMNIST() function courtesy of https://github.com/sunsided/mnist-matlab

convertMNIST()
load('mnist.mat')

%% Take a 10% selection for building code

% Select random indices
[miniTrainIdx,miniValIdx,~] = dividerand(size(training.images,3),0.1,0.03,0.87);
[miniTestIdx,~,~] = dividerand(size(test.images,3),0.1,0.45,0.45);

% Separate the training data
miniTrain = cell(size(miniTrainIdx,2),2);
for i = 1:size(miniTrain,1)
    miniTrain{i,1} = training.images(:,:,miniTrainIdx(i));
    miniTrain{i,2} = training.labels(miniTrainIdx(i));
end

% Separate the validation data
miniVal = cell(size(miniValIdx,2),2);
for i = 1:size(miniVal,1)
    miniVal{i,1} = training.images(:,:,miniValIdx(i));
    miniVal{i,2} = training.labels(miniValIdx(i));
end

% Separate the test data and create a 4D array for testing
miniTest = cell(size(miniTestIdx,2),2);
miniTest4D = double.empty(28,28,1,0);
for i = 1:size(miniTest,1)
    miniTest{i,1} = test.images(:,:,miniTestIdx(i));
    miniTest{i,2} = test.labels(miniTestIdx(i));
    miniTest4D = cat(4,miniTest4D,test.images(:,:,miniTestIdx(i)));
end

% Reformat to tables
miniTrain = cell2table(miniTrain, 'VariableNames',{'X','y'});
miniTrain.y = categorical(miniTrain.y);

miniVal = cell2table(miniVal, 'VariableNames',{'X','y'});
miniVal.y = categorical(miniVal.y);

miniTest = cell2table(miniTest, 'VariableNames',{'X','y'});
miniTest.y = categorical(miniTest.y);

% Save data
save('miniMNIST.mat','miniTest','miniTest4D','miniTrain','miniVal');
%% Organise the full data set for use

% Select random indices
[trainIdx,valIdx,~] = dividerand(size(training.images,3),0.7,0.3,0);
testIdx = 1:size(test.images,3);

% Separate the training data
train = cell(size(trainIdx,2),2);
for i = 1:size(train,1)
    train{i,1} = training.images(:,:,trainIdx(i));
    train{i,2} = training.labels(trainIdx(i));
end

% Separate the validation data
val = cell(size(valIdx,2),2);
for i = 1:size(val,1)
    val{i,1} = training.images(:,:,valIdx(i));
    val{i,2} = training.labels(valIdx(i));
end

% Separate the test data and create a 4D array for testing
testCell = cell(size(testIdx,2),2);
test4D = double.empty(28,28,1,0);
for i = 1:size(testCell,1)
    testCell{i,1} = test.images(:,:,testIdx(i));
    testCell{i,2} = test.labels(testIdx(i));
    test4D = cat(4,test4D,test.images(:,:,testIdx(i)));
end

% Reformat to tables
train = cell2table(train, 'VariableNames',{'X','y'});
train.y = categorical(train.y);

val = cell2table(val, 'VariableNames',{'X','y'});
val.y = categorical(val.y);

testCell = cell2table(testCell, 'VariableNames',{'X','y'});
testCell.y = categorical(testCell.y);

save('Full_MNIST_Arranged.mat','testCell','test4D','train','val');
%% Create Final Model (Conv layers = 4; Filters/layer = 64)

% Training Options
options = trainingOptions('sgdm',...
    'MaxEpochs',5, ...
    'ValidationData',val,...
    'ValidationFrequency',205,...
    'ValidationPatience',4,...
    'Verbose',false,...
    'Plots','training-progress');

batchNormalisation = true;
maxPooling = true;
convLayers = 4;
numFilters = 64;

finalNetResults = buildTrainTestCNN(convLayers,numFilters,...
    batchNormalisation,maxPooling,options,train,test4D,testCell);

save finalNetResults
