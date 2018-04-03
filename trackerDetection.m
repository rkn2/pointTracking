%setup 
imageFolder = 'G:\My Drive\Documents\Research\mikehess\paper1_baptistery\computervision\';
circleImageLocation = strcat(imageFolder,'target.jpg');
circleImage = imread(circleImageLocation);
circleImage = rgb2gray(circleImage);
%figure; 
%imshow(circleImage);
sceneImageLocation = strcat(imageFolder,'brick.jpg');
sceneImage = imread(wallImageLocation);
sceneImage = rgb2gray(sceneImage);
%figure; 
%imshow(sceneImage);

%now detect the features
circlePoints = detectSURFFeatures(circleImage);
scenePoints = detectSURFFeatures(sceneImage);

% figure;
% imshow(circleImage);
% hold on;
% plot(selectStrongest(circlePoints, 100));
% 
% figure;
% imshow(sceneImage);
% hold on;
% plot(selectStrongest(scenePoints, 100));

%extraact feature descriptors
[circleFeatures, circlePoints] = extractFeatures(circleImage, circlePoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage, scenePoints);

%find putative point matches
circlePairs = matchFeatures(circleFeatures, sceneFeatures);

matchedCirclePoints = circlePoints(circlePairs(:, 1), :);
matchedScenePoints = scenePoints(circlePairs(:, 2), :);

[tform, inlierBoxPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedCirclePoints, matchedScenePoints, 'affine');


circlePolygon = [1, 1;...                           % top-left
        size(circleImage, 2), 1;...                 % top-right
        size(circleImage, 2), size(circleImage, 1);... % bottom-right
        1, size(circleImage, 1);...                 % bottom-left
        1, 1];                   % top-left again to close the polygon
    
newCirclePolygon = transformPointsForward(tform, circlePolygon);

figure;
imshow(sceneImage);
hold on;
line(newCirclePolygon(:, 1), newCirclePolygon(:, 2), 'Color', 'y');
title('Detected Target');