%%DRAW BRICKS
function draw_bricks()
close all
figure

%use global because we want black filled in sections

%% create binary image
%rgb = im2double(imread('test.png'));
rgb = im2double(imread('whiteBrick.jpg'));
%rgb = im2double(imread('Global.jpg'));
%rgb = im2double(imread('simCorrect.png'));
gs = max(rgb,[],3); %make it grayscale
bw = gs < 0.67; %this inverts what is black and what is white. 
bw = bwareaopen(bw, 500); %get rid of objects with < 33 pixels in them

bw_orig = imcomplement(bw);

%morphologically close image
se = strel('square',50);
%se = strel('line',100,0);
bw = imclose(bw,se);

bw = imcomplement(bw); %turn it back into white bricks and black background

im = imshow(bw);
uistack(im,'bottom') % analogous with "send to back" in ppt
hold on

%% set up grid for easily referencing pixels later
[gX,gY] = meshgrid(1:size(bw,2),1:size(bw,1));
grid = [reshape(gX,[],1),reshape(gY,[],1)]; 

%% find corners
area_thresh = 500;
threshold = 1;
pca_scale = 10;
% this works well for regulated blocks 
% but I am a bit worried about the physical wall
corners = detectMinEigenFeatures(bw); %built in CV to get corners
corner_points = double(corners.selectStrongest(1000).Location); % x and y 
% plot(corner_points(:,1),corner_points(:,2),'g.');
[pts,c] = voronoin([corner_points(:,2),corner_points(:,1)]);
pts(sum(pts > size(bw),2)>0,:) = []; %checks bounds of voroy in image
pts(sum(pts < 1,2)>0,:) = []; %set bad ones to blank
for idx = 1:length(pts)
    p = round(pts(idx,:)); %bc pixels are ints but voroy can give floats
    %black = 0; white = 1 
    if( ~bw(p(1),p(2)) ) % ~ means not, if pixel is black 
        %plot(p(2),p(1),'g.','markersize',6)
        continue % dont start a brick from a black space
    else
        %plot(p(2),p(1),'b.','markersize',6) %plot where brick is starting
    end
    theta = 0:pi/128:2*pi; %start, increment, end; series of angles
    theta = theta(1:(end-1)); %removes 2pi bc redundant 
    r = zeros(length(theta),2); %takes notes about collisions
    for t = 1:length(theta)
        r(t,:) = extent(bw,p,theta(t),threshold); % pass image, start point, angle return first black pixel found
    end
    s = fliplr(r); %bc images are plotted backwards
    if( polyarea(s(:,1),s(:,2)) < area_thresh )
        continue
    end
    %plot(s(:,1),s(:,2),'bo')
    h = filter_hull(s); %spits out edges given black points 
    %plot(h(:,1),h(:,2),'b-','linewidth',1) %plot the hull
    polyin = polyshape(h(:,:),'Simplify',false); %defines polygon so we can get centroid
    [cx, cy] = centroid(polyin); %get centroid coordinates
    %plot(cx,cy,'rx','markersize',16) % plot centroid
    % start over casting rays from centroid
    r = zeros(length(theta),2); %takes notes about collisions
    for t = 1:length(theta)
        r(t,:) = extent(bw,[cy,cx],theta(t),threshold); % pass image, start point, angle return first black pixel found
    end
    s = fliplr(r);
    %plot(s(:,1),s(:,2),'rs')
    h = filter_hull(s); %spits out edges given black points 
    polyin = polyshape(h(:,:),'Simplify',false); %defines polygon so we can get centroid
    [cx, cy] = centroid(polyin); %get centroid coordinates
    plot(cx,cy,'ro','markersize',16) % plot centroid
    plot(h(:,1),h(:,2),'r-','linewidth',1) %plot the hull
    coeff = pca(h);
    plot(cx+coeff(1,1)*[-1,+1]*pca_scale,cy+coeff(2,1)*[-1,+1]*pca_scale,'r-');
    plot(cx+coeff(1,2)*[-1,+1]*pca_scale,cy+coeff(2,2)*[-1,+1]*pca_scale,'r-');
    % pause
    f = inhull(grid,h(:,:)); %returns a logical array of pixels inside the hull/polygon
    %so 0 and 1 for every pixel
    bw(f) = 0; %any pixel where f is true gets set to 0; 0 = black 
    im.CData = bw;
    % c = find_corners(r(h,:));
    % plot(c(:,1),c(:,2),'g-','linewidth',2)
    % plot(mean(c(1:4,1)),mean(c(1:4,2)),'g+','linewidth',2)
    drawnow
    %pause
end
im.CData = bw_orig;
drawnow
end
%%
function out = extent(bw,p,theta,threshold)
v = [cos(theta),sin(theta)]; %tells you the vector youre looking on
l = 0; %chnges length of vector youre looking on
x = round(p + v * l); %furthest point you can see; only pt you see for now
c = 0;
while(c < threshold)
    c = c + ~bw(x(1),x(2)); % tally if pixel is black
    l = l + 1; %trying to find a black pixel
    x = round( p + v * l ); %new pixel we are looking at
    if( any( x < 1 ) ) %bounds to stay inside the image
        x = max(x,[1 1]);
        break;
    end
    if( any( x > size(bw) ) ) %still bounds
        x = min(x,size(bw));
        break;
    end
end
out = x; %returns the first place we found a black pixel
end

function [d,r] = point_to_line(q, p1, p2) %for old version
a = p2 - p1;
b = q - p1;
d = norm(cross(a,b)) / norm(a);
r = dot(a,b)/dot(a,a)*a + p1;
end

function out = tri_area(a,b,c)
out = 0.5 * abs( (a(1)-c(1))*(b(2)-a(2))-(a(1)-b(1))*(c(2)-a(2)) );
end

function out = filter_hull(outline)
h = convhull(outline);
out = outline(h,:);
polyin = polyshape(outline(h,:),'Simplify',false);
[cx, cy] = centroid(polyin);
%plot(cx,cy,'bs')
A0 = polyarea(outline(h,1),outline(h,2));
f = zeros(length(outline),1);
half_w = 5;
for i = 1:length(outline)
    in = 1:length(outline);
    for j = (i-half_w):(i+half_w)
        k = j;
        if( k < 1 )
            k = length(outline) + k;
        end
        if( k > length(outline) )
            k = k - length(outline);
        end
        in(k) = NaN;
    end
    in(in~=in) = [];
    inh = convhull(outline(in,:));
    f(i) = polyarea(outline(in(inh),1),outline(in(inh),2)) / A0;
end
dev = f - mean(f);
stdev = sqrt(var(dev));
z = dev / stdev;
if( ( any(z<-3) && min(f) < 0.85 ) || min(f) < 0.67 )
    %fprintf('outlier(s) detected!\n')
    ok = find(z>-3);
    h = convhull(outline(ok,:));
    out = outline(ok(h),:);
    %plot(out(:,1),out(:,2),'gs-')
    %keyboard
end
end

function out = find_corners(r) %old version
w = 3;
theta = zeros(length(r),w);
dists = zeros(length(r),w);
for j = 1:w
    for i = 1:(length(r)-j)
        x = r(i+j,:)-r(i,:);
        theta(i,j) = atan2(x(2),x(1));
        dists(i,j) = sqrt(sum(x.^2));
    end
    for i = 1:j
        x = r(i,:)-r(end-(j-i),:);
        theta(end-(j-i),j) = atan2(x(2),x(1));
        dists(end-(j-i),j) = sqrt(sum(x.^2));
    end
end
[hy,hx] = hist(reshape(abs(theta),[],1),32);
theta_major = hx(find(hy==max(hy)));
hz = hy;
hz(abs(hx-theta_major)<pi/4) = 0;
theta_minor = hx(find(hz==max(hz)));
lines = zeros(4,2);
% convert points on major axis to lines
dt = abs(theta(:,1))-theta_major;
dt = dt - pi*round(dt/pi);
r_major = find(abs(dt)<pi/8);
r_major(diff(r_major)>ceil(length(r)/10)) = [];
Z = linkage(r(r_major,:));
c = cluster(Z,'maxclust',2);
lines(1,:) = assign_line(r(r_major(c==1),:));
lines(2,:) = assign_line(r(r_major(c==2),:));
%{
plot(r(r_major(c==1),2),r(r_major(c==1),1),'b+')
plot(r(r_major(c==1),2),polyval(lines(1,:),r(r_major(c==1),2)),'b-')
plot(r(r_major(c==2),2),r(r_major(c==2),1),'b+')
plot(r(r_major(c==2),2),polyval(lines(2,:),r(r_major(c==2),2)),'b-')
%}
% repeat with minor axis
dt = abs(theta(:,1))-theta_minor;
dt = dt - pi*round(dt/pi);
r_minor = find(abs(dt)<pi/8);
r_minor(diff(r_minor)>ceil(length(r)/10)) = [];
Z = linkage(r(r_minor,:));
c = cluster(Z,'maxclust',2);
lines(3,:) = assign_line(r(r_minor(c==1),:));
lines(4,:) = assign_line(r(r_minor(c==2),:));
%{
plot(r(r_minor(c==1),2),r(r_minor(c==1),1),'c+')
plot(r(r_minor(c==1),2),polyval(lines(3,:),r(r_minor(c==1),2)),'c-')
plot(r(r_minor(c==2),2),r(r_minor(c==2),1),'c+')
plot(r(r_minor(c==2),2),polyval(lines(4,:),r(r_minor(c==2),2)),'c-')
%}
% find intersections between edge lines
corners = zeros(4,2);
pairs = [1,3;1,4;2,3;2,4];
for i = 1:4
    corners(i,:) = intersect(lines(pairs(i,1),:),lines(pairs(i,2),:));
    if( any(isnan(corners(i,:))) )
        two_lines = [lines(pairs(i,1),:);lines(pairs(i,2),:)];
        fit_line = two_lines(~isnan(sum(two_lines,2)),:);
        nan_line = two_lines(isnan(sum(two_lines,2)),:);
        fit_dim = find(~isnan(nan_line));
        nan_dim = find(isnan(nan_line));
        corners(i,nan_dim) = nan_line(fit_dim);
        corners(i,fit_dim) = polyval(fit_line,nan_line(fit_dim));
    end
end
c = convhull(corners);
out = corners(c,:);
end

function out = assign_line(x) % is this used?
if(any(range(x)<1))
    out = mode(x);
    out(range(x)>1) = NaN;
else
    out = polyfit(x(:,2),x(:,1),1);
end
end

function out = intersect(p,q) % is this used?
a = p(1);
b = q(1);
c = p(2);
d = q(2);
x = ( d - c) / ( a - b );
y = a * x + c;
out = [x,y];
end