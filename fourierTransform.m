fullFileName = 'G:\My Drive\Documents\Research\mikehess\paper1_baptistery\computervision\wall2.jpg';
rgb = im2double(imread(fullFileName));
gs = max(rgb,[],3);
gs = gs<0.33;
imshow(gs)

%reg = [1287.5 831.5 891 388]; wall.jpg
reg = round([1791.5 420.5 480 141]);
a = gs(reg(2): reg(2) + reg(4), reg(1): reg(1) + reg(3));
C = real(ifft2(fft2(gs) .* fft2(rot90(a,2),size(gs,1), size(gs,2))));
figure;
imshow(C,[]); % Scale image to appropriate display range.

max(C(:))

thresh = 2000; % Use a threshold that's a little less than max.
D = C > thresh;
se = strel('disk',5);
E = imdilate(D,se);
figure
imshow(E) % Display pixels with values over the threshold.
