%% Clean DataSet
clear;close all;clc
datasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
images = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
sp = randperm(10000,25);
figure;
for i = 1:25
    subplot(5,5,i);
    imshow(images.Files{sp(i)});
end
for i=1:length(images.Files)
    img = readimage(images,i);
    m = size(img);
    if m>28
        error('Reshape images to the correct size');
    end
end
label_Count = countEachLabel(images)
numIteration = 350;
[TrainImages,ValidationImages] = splitEachLabel(images,numIteration,'randomize');
layers = [
    imageInputLayer([size(img,1) size(img,2) 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
%     fullyConnectedLayer(2048)
%     reluLayer
%     dropoutLayer
    
    fullyConnectedLayer(40)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam',...
    'InitialLearnRate',0.002,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'MaxEpochs',10,...
    'MiniBatchSize',100,...
    'ValidationData',ValidationImages,...
    'Plots','training-progress',...
    'OutputFcn',@(info)savetrainingplot(info));
%@(info)stopIfAccuracyNotImproving(info,3),
[net , info]= trainNetwork(TrainImages,layers,options);
YPredicted = classify(net,ValidationImages);
YValidation = ValidationImages.Labels;
accuracy = sum(YPredicted == YValidation)/numel(YValidation)

layerss = [2 6 10];
channels = 1:8;
for layer = layerss
    I = deepDreamImage(net,layer,channels,...
        'Verbose',false, ...
        'PyramidLevels',1);

    figure
    montage(I)
    name = net.Layers(layer).Name;
    title(['Layer ',name,' Features'])
end

layers_Name = [
            imageInputLayer([28 28 1], 'Name', 'Image input')
            convolution2dLayer(3, 8, 'Name', 'conv_1')
            batchNormalizationLayer('Name','batchnorm_1')
            reluLayer('Name', 'ReLU_1')
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
            convolution2dLayer(3, 16, 'Padding', 1, 'Name', 'conv_2')
            batchNormalizationLayer('Name','batchnorm_2')
            reluLayer('Name', 'ReLU_2')
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
            convolution2dLayer(3, 32, 'Padding', 1, 'Name', 'conv_3')
            batchNormalizationLayer('Name','batchnorm_3')
            reluLayer('Name', 'ReLU_3')
            fullyConnectedLayer(40, 'Name', 'fullyconnected')
            fullyConnectedLayer(10, 'Name', 'fullyconnected2')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'classoutput')];
        
lgraph = layerGraph(layers_Name);
figure
plot(lgraph)
%% Noisy DataSet
% imgs = readall(images);
% m=size(imgs,1);
% noisyimg= cell(1,m);
% for i=1:m
% noisyimg(i) = {imnoise(imgs{i,1},'gaussian')};
% filename = ['' num2str(i) '.png'];
% x=noisyimg(i);
% imwrite(x{1,1},filename);
% end
% noisyimg=noisyimg';
% for i=9001:10000
%     if i==10001
%         break
%     end
%     filename = ['' num2str(i) '.png'];
%     movefile(filename,'9','f')
% end
datasetPathNoisy = fullfile('C:\Users\soufi\Desktop\My Courses\Semester 5 - SS2018\project');
imagesNoisy = imageDatastore(datasetPathNoisy, ...
    'IncludeSubfolders',true,'FileExtensions','.png','LabelSource','foldernames');
sp = randperm(10000,25);
figure;
for i = 1:25
    subplot(5,5,i);
    imshow(imagesNoisy.Files{sp(i)});
end
for i=1:length(imagesNoisy.Files)
    img = readimage(imagesNoisy,i);
    m = size(img);
    if m>28
        error('Reshape images to the correct size');
    end
end
label_Count = countEachLabel(imagesNoisy)
numIteration = 350;
% numIteration = 800;
[TrainImages,ValidationImages] = splitEachLabel(imagesNoisy,numIteration,'randomize');
layers = [
    imageInputLayer([size(img,1) size(img,2) 1])
    
    convolution2dLayer(3,8,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
    
    fullyConnectedLayer(40) %2048
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam',...
    'InitialLearnRate',0.002,...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.999,...
    'Epsilon',1e-8,...
    'MaxEpochs',10,...
    'MiniBatchSize',100,...
    'ValidationData',ValidationImages,...
    'Plots','training-progress',...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));
%     'L2Regularization',0.003,...

[net , info]= trainNetwork(TrainImages,layers,options);
YPredicted = classify(net,ValidationImages);
YValidation = ValidationImages.Labels;
accuracy = sum(YPredicted == YValidation)/numel(YValidation)
%accuracy is 97.35 with no L2 Regularizer 
%accuracy is 98.28 with L2 Regularizer 0.0001 & 0.001
%accuracy is 98.71 with L2 Regularizer 0.003
% numIteration = 800 no change accuracy stayed at 98.70 
%97.42 more neurons at first layer dropout didn't have any effect