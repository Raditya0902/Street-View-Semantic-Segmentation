pretrainedNetwork = fullfile('D:\Programming\MATLAB\SemanticSegmentation','deeplabv3plusResnet18CamVid.mat');  
data = load(pretrainedNetwork);
net = data.net;
%%
classes = string(net.Layers(end).Classes);
%%
I = imread('highway.png');
%%
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));
%%
C = semanticseg(I,net);
%%
cmap = camvidColorMap;
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);
%%
resnet18();
%%
imgDir = fullfile("D:\Programming\MATLAB\SemanticSegmentation\701_StillsRaw_full");
imds = imageDatastore(imgDir);
%%
I = readimage(imds,559);
I = histeq(I);
imshow(I);
%%
classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];
%%
labelIDs = camvidPixelLabelIDs();
%%
labelDir = fullfile('D:\Programming\MATLAB\SemanticSegmentation\PixelingImages', 'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
%%
C = readimage(pxds,559);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
imshow(B)
pixelLabelColorbar(cmap,classes);
%%
tbl = countEachLabel(pxds);
%%
frequency = tbl.PixelCount/sum(tbl.PixelCount);

bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
%%
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);
%%
numTrainingImages = numel(imdsTrain.Files);
%%
numValImages = numel(imdsVal.Files);
%%
numTestingImages = numel(imdsTest.Files);
%%
% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [720 960 3];

% Specify the number of classes.
numClasses = numel(classes);

% Create DeepLab v3+.
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");
%%
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
%%
pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph, 'classification', pxLayer);
%%
% Define validation data.
dsVal = combine(imdsVal,pxdsVal);

% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',30, ...  
    'MiniBatchSize',4, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', 'D:\Programming\MATLAB\SemanticSegmentation\checkPoints', ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4);
%%
dsTrain = combine(imdsTrain, pxdsTrain);
%%
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data,xTrans,yTrans));
%%
doTraining = false;
if doTraining    
    [net, info] = trainNetwork(dsTrain,lgraph,options);
end
%%
load('trained_network_semantic_segmentation.mat', 'net');
%%
I = readimage(imdsTest,35);
C = semanticseg(I, net);
%%
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);
%%
expectedResult = readimage(pxdsTest,35);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected);
%%
iou = jaccard(C,expectedResult);
table(classes,iou)
%%
pxdsResults = semanticseg(imdsTest,net, ...
    'MiniBatchSize',4, ...
    'WriteLocation','D:\Programming\MATLAB\SemanticSegmentation\PixelingImages\PixelingResults', ...
    'Verbose',false);
%%
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest,'Verbose',false);
%%
metrics.DataSetMetrics
%%
metrics.ClassMetrics
%%
save("trained_network_semantic_segmantation.mat", 'net');