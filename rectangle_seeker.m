image = 'G:\My Drive\Documents\Research\mikehess\paper1_baptistery\computervision\wall.jpg';
rgb = imread(image);
bw = im2bw(rgb);
figure
imshow(rgb)

%%% does not work

% stats = [regionprops(bw);regionprops(not(bw))];
% imshow(bw);
% hold on;
% for i = 1:numel(stats)
%     rectangle('Position', stats(i).BoundingBox, ...
%     'Linewidth', 3, 'EdgeColor', 'r', 'LineStyle', '--');
% end

%%% harris-stephens algorithm does not work, it did not get any corners
% corners = detectHarrisFeatures(bw, 'ROI', [100 300 3900 2700]);
% imshow(bw); 
% hold on;
% plot(corners.selectStrongest(50));



%%%use Hu's moment
regions = [1574, 713;1580, 839;2024, 845;2009, 710];
xMin = 1570;
yMin = 705;
width = 450;
height = 140;
areaRegion = (width * height)-1000;
regionSize = size(regions);
matchIndex = 1;
croppedImage = imcrop(bw, [xMin yMin width height]);
figure;
imshow(croppedImage)



for i = 1:regionSize(1)
     %croppedImage = imcrop(bw, regions);          
     mask = bwareaopen(croppedImage, areaRegion);
     %Compute Hu's moment 
     hu = invmoments(mask);
     tol = 0.1;
       LIA = ismembertol(hu,croppedImage,tol);
       if(mean(LIA) > 0.9)
           matches(matchIndex) = regions(i);
           matchIndex =matchIndex + 1;
       end
  end