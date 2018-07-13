%% Heatmaps for Accuracy and Time
load('FullResults.mat','results');
close all;
data = results(5:end,[1 2 7 8]);
data.convLayers = cell2mat(data.convLayers);
data.numFilters = cell2mat(data.numFilters);
data.testAccuracy = cell2mat(data.testAccuracy);
data.TrainingTime = cell2mat(data.TrainingTime);

figure;
h1 = heatmap(data,'numFilters','convLayers','ColorVariable','testAccuracy');
h1.Title = 'Test Data Accuracy';
h1.XLabel = 'Filters per Layer';
h1.YLabel = 'Convolutional Layers';
h1.Colormap = parula;
figure;
h2 = heatmap(data,'numFilters','convLayers','ColorVariable','TrainingTime');
h2.Title = 'Training Time';
h2.XLabel = 'Filters per Layer';
h2.YLabel = 'Convolutional Layers';
h2.Colormap = parula;

%% Surface Plot Accuracy
data = results(5:end,[1 2 7]);
data.convLayers = cell2mat(data.convLayers);
data.numFilters = cell2mat(data.numFilters);
data.testAccuracy = cell2mat(data.testAccuracy);
data = unstack(data,'testAccuracy','convLayers');
X = 1:5;
Y = [16 32 64 128];

figure;
surf(X,Y,table2array(data(:,2:end)));
ylabel('Filters per Layer');
xlabel('Convolutional Layers');
zlabel('Test Accuracy');

%% Surface Plot Time
data = results(5:end,[1 2 8]);
data.convLayers = cell2mat(data.convLayers);
data.numFilters = cell2mat(data.numFilters);
data.TrainingTime = cell2mat(data.TrainingTime);
data = unstack(data,'TrainingTime','convLayers');
X = 1:5;
Y = [16 32 64 128];

figure;
surf(X,Y,table2array(data(:,2:end)));
ylabel('Filters per Layer');
xlabel('Convolutional Layers');
zlabel('Training Time');

%% Filter Visualisation
load('CNN_L4_F64_BNT_MPT.mat','net');
% List layers
net.Layers
i=1;
figure;
for layer = [2 5 9 13]
    % Set Layer of interest
    current_layer = layer;
    name = net.Layers(current_layer).Name;
    % Set which filters in layer to view
    channels = 1:64;
    I = deepDreamImage(net,current_layer,channels, ...
        'PyramidLevels',1);
    % View
    subplot(2,2,i)
    montage(I)
    title(['Layer ',name,' Features'])
    i = i+1;
end

%% Examples of incorrectly classified images

load('Full_MNIST_Arranged.mat','testCell','test4D','train','val');
% Test it (with timing)
tic;
yhat = classify(net,test4D);
testTime = toc;
testAccuracy = sum(yhat == testCell.y)/numel(testCell.y);
% Identify and separate incorrectly classified images
yhat = table(yhat);
testCell = [testCell yhat];
incorrect = testCell(testCell.y ~= testCell.yhat,:);
% View them
figure;
montage(incorrect.X)
title('Incorrectly Classified Images')
save('incorrect.mat', 'incorrect')

figure;
for row = 1:size(incorrect,1)
    subplot(8,9,row);
    imshow(incorrect.X{row});
    % Adds label above image showing 'truth -> model prediction'
    title(strcat(char(incorrect.y(row)),'->',char(incorrect.yhat(row))));
end