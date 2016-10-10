function div=divergence(fx,fy)

gx = fx-fx([1 1:end-1],:);
gx(1,:)   = fx(1,:);
gx(end,:) = -fx(end-1,:);
gy = fy-fy(:,[1 1:end-1]);
gy(:,1)   = fy(:,1);
gy(:,end) = -fy(:,end-1);

div=gx+gy;