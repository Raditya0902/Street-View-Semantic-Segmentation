%%
load('trained_network_semantic_segmentation.mat', 'net');
%%
imgDir = fullfile("D:\Programming\MATLAB\SemanticSegmentation\701_StillsRaw_full");
imds = imageDatastore(imgDir);
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
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);
%%
I = readimage(imdsTest,35);
C = semanticseg(I, net);
%%
cmap = camvidColorMap;
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