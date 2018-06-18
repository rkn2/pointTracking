function draw_normals%(file, caxis)
files = dir('simulations/cManual/*fVertex.dat'); %read in final vertex files
filename = ['simulations/cManual/',files(1).name]; %get the names of the files, only does one for now

min_gap = 0.002;
vecscale = 0.054;

X = readVerts(filename); %read in verts
bid = X(:,1);
Vf = X(:,2:4);
% length(Vf)
% length(unique(Vf(:,1)))

front = find(Vf(:,2) > 0.05); %find where the y is greater than 0.05 to get only the front half
Vf = Vf(front,[1,3]); %get final front vertices in terms of x and z (getting rid of out of plane) (x, z)
%Vf = unique(Vf,'rows'); THIS ALSO DOES NOT WORK
bid = bid(front);

%nonbase = 1:260; %d
nonbase = 1:300; %c
%nonbase = 1:328; %this is only for b
%nonbase = 1:372; %this is only for A % the rest are 'base' blocks 
Vf = Vf(nonbase,:); %x,z
bid = bid(nonbase);

figure;
%plot(Vf(:,1),Vf(:,2),'k.')
hold on
axis equal

nb = length(unique(bid));



bricks = {};
brick_faces = {};
perimeters = {};
centroids = zeros(nb,2);
for i = 1:nb
    idx = find(bid == i); % find the blocks that match this identifier
    centroids(i,:) = mean(Vf(idx,:),1);
    perim = zeros(length(idx)+1,2); % allocate array for plotting the perimeter
    perim_order = convhull(Vf(idx,:)); % use convhull to find the perimeter ordering
    perim = Vf(idx(perim_order),:); % assign the perimeter
    bricks{i} = polyshape(perim);
    perimeters{i} = perim; % assign brick perimeter coordinates in order
    
    line = plot(perim(:,1),perim(:,2),'k-');
    
    normals = normalsFromPerim(perim); % get the normals from our helper function
    
    faces = {};
    for j = 1:length(idx) %j is the row number
        face = [perim(j,:);
            perim(j+1,:);
            perim(j+1,:)+normals(j,:)*vecscale;
            perim(j,:)+normals(j,:)*vecscale];
        faces{j} = polyshape(face);
        %plot(faces{j})
    end
    brick_faces{i} = faces;
end

%now that we have all of the edges and their boundaries, go back and look
%for cracks

rad = vecscale * 2;
num_rays = 10;
crackscale = 5;

cx = [];
cy = [];
cc = [];
brick_cracks = {};
for i = 1:nb
    perim = perimeters{i};
    normals = normalsFromPerim(perim);
    cracks = zeros(length(normals),1);
    adj = zeros(length(Vf),1); %adj has as entry for each vertex
    %d_self = pdist(perimeters{i});
    %rad = max(d_self);
    for j = 1:(length(perimeters{i})-1) %minus one because redundant
        v = Vf - perimeters{i}(j,:).*ones(size(Vf)); % vectors between this vert and all other verts
        d = sqrt(sum(v.^2,2)); % magnitudes of those displacement vectors
        adj(d<rad) = 1; % mark verts within the radius
    end
    nbr_bricks = unique(bid(adj==1));
    faces = brick_faces{i};
    for j = 1:length(faces)
        hit = zeros(length(nbr_bricks),1);
        for k = 1:length(nbr_bricks)
            hit(k) = overlaps(faces{j},bricks{nbr_bricks(k)});   
        end
        hit_bricks = nbr_bricks(hit==1);
        best_d = ones(num_rays,1) * NaN;
        best_v = ones(num_rays,2) * NaN;
        for k = 1:num_rays
            ray0 = interpAlong(perim(j,:),perim(j+1,:),k/(num_rays+1)) - normals(j,:) * 0.5 * vecscale; % backward ray
            ray1 = ray0 + normals(j,:) * 0.5 * vecscale; % actual face
            ray2 = ray1 + normals(j,:) * vecscale; % forward ray
            ray = [ray0;ray1;ray2];
            %plot(ray(:,1),ray(:,2)) %%%
            for l = 1:length(hit_bricks)
                hb = perimeters{hit_bricks(l)};
                [x,y] = polyxpoly(ray(:,1),ray(:,2),hb(:,1),hb(:,2));
                if( ~isempty(x) && ~isempty(y) ) % ray hit the brick
                    % find nearest hit
                    hitvecs = [x,y] - ones(size(x,1),1)*ray1;
                    d = sqrt(sum( hitvecs.^2, 2 ));
                    mdx = find(d==min(d));
                    if( d(mdx) < best_d(k) || isnan(best_d(k)) )
                        best_d(k) = d(mdx);
                        best_v(k,:) = hitvecs(mdx,:);
                    end
                end
            end
            %plot(best_v(k,1),best_v(k,2),'k-') %%%
        end
        bd = [0,0;transpose(1:(num_rays))/(num_rays+1),best_d];
        too_small = find( bd(:,2) < min_gap );
        bd(too_small,:) = [];
        cracks(j) = nanmean(bd(:,2));
        sc = cumsum(bd(:,2)) / nansum(bd(:,2));
        ok = find(~isnan(sc));
        ok_start = find(sc(ok)>0,1);
        if( ok_start > 1 )
            ok_start = ok_start - 1;
        end
        ok_end = find(sc(ok)==1,1);
        ok = ok(ok_start:ok_end);
        if( sum(ok) == 0 )
            continue
        end
        if( sum(ok) == 1 )
            pos = (find(ok)-1)/(length(ok)-1);
        else
            pos = interp1(sc(ok),bd(ok,1),0.5);
        end
        v0 = interpAlong(perim(j,:),perim(j+1,:),pos);
        v1 = v0 + normals(j,:) * cracks(j) * crackscale;
        vc = [v0;v1];
        cx = [cx;vc(:,1);nan];
        cy = [cy;vc(:,2);nan];
        cc = [cc;cracks(j);cracks(j);0];
    end
    brick_cracks{i} = cracks;
end
h = surface('XData',[cx(:) cx(:)],'YData',[cy(:) cy(:)],...
            'ZData',zeros(length(cx(:)),2),'CData',[cc(:) cc(:)],...
            'FaceColor','none','EdgeColor','flat','Marker','none',...
            'LineWidth',4);

%colorbar
colormap('jet')
caxis([0.001 0.025])
figureName = ['simulations/cManual/',files(1).name,'.pdf'];
axis off
saveas(gcf,figureName);
end

function normals = normalsFromPerim(perim)
ne = (length(perim)-1); % number of edges
normals = zeros(ne,2); % allocate array for normals
for i = 1:ne % for each edge defining the perimeter
    edge = perim(i+1,:) - perim(i,:); % calculate the edge vector
    R = rotMat2D(pi/2); % rotate the edge vector 90 degrees to get normal
    normals(i,:) = edge * R;
    normals(i,:) = normals(i,:) / sqrt(sum(normals(i,:).^2));
end
end

function out = midpointsFromPerim(perim)
ne = (length(perim)-1); % number of edges
out = zeros(ne,2); % allocate array for normals
for i = 1:ne % for each edge defining the perimeter
    out(i,:) = interpAlong(perim(i,:),perim(i+1,:),0.5);
end
end

function r = interpAlong(p,q,f)
r = zeros(length(f),length(p));
for i = 1:length(p)
    r(:,i) = interp1([0,1],[p(i),q(i)],f);
end
end

function R = rotMat2D(theta)
R = [cos(theta),-sin(theta);sin(theta),cos(theta)];
end

% function [d,r] = pointToLine(q, p1, p2)
% a = p2 - p1;
% b = q - p1;
% d = norm(cross(a,b)) / norm(a);
% r = dot(a,b)/dot(a,a)*a + p1;
% end
% 
% function out = lineIntersection(line1,line2)
% slope = @(line) (line(2,2) - line(1,2))/(line(2,1) - line(1,1));
% intercept = @(line,m) line(1,2) - m*line(1,1);
% m1 = slope(line1);
% m2 = slope(line2);
% b1 = intercept(line1,m1);
% b2 = intercept(line2,m2);
% xintersect = (b2-b1)/(m1-m2);
% yintersect = m1*xintersect + b1;
% isPointInside = @(xint,myline) ...
%     (xint >= myline(1,1) && xint <= myline(2,1)) || ...
%     (xint >= myline(2,1) && xint <= myline(1,1));
% inside = isPointInside(xintersect,line1) && ...
%          isPointInside(xintersect,line2);
% out = [xintersect,yintersect];
%{
lineEq = @(m,b, myline) m * myline(:,1) + b;
yEst2 = lineEq(m1, b1, line2);
enderr = @(ends,line) ends - line(:,2);
errs1 = enderr(yEst2,line2);
yEst1 = lineEq(m2, b2, line1);
errs2 = enderr(yEst1,line1);
% possibleIntersection =  sum(sign(errs1))==0 && sum(sign(errs2))==0;
 %}
% end