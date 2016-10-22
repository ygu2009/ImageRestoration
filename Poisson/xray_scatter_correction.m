% Author: Yingying Gu (ying.y.gu@gmail.com)
% version 1.0
% Copyright 2016
% University of Wisconsin-Milwaukee
% Project: Scatter Reduction for X-ray Image

clc
clear
close all

load bone.mat

% deblurring kernel
sigma_1 = 1.5;
sigma_2 = 1.5;
win_size = 11;
[x y]=meshgrid(round(-win_size/2):round(win_size/2), round(-win_size/2):round(win_size/2));
gauss2DFilter=exp(-x.^2/(2*sigma_1^2)-y.^2/(2*sigma_2^2));
gauss2DFilter=gauss2DFilter./sum(gauss2DFilter(:));

H=gauss2DFilter;

alpha=1;
beta=1;

figure,imshow(raw,[]), title('Scatter Image');

% calculate the gradient of 2D image
gradfx = @(x)x([2:end end],:)-x;
gradfy = @(x)x(:,[2:end end])-x;

[M,N]=size(raw);


%% Deconvolution by Total Variation (TVL1) Regularization
lambda = 0.001;
epsilon = 1*1e-4;
tau = 0.005;
u = raw;

% Using the iterative Gradient Descent method to minimize the cost function
iter=1;
while iter<1000
    iter=iter+1;
    ux=gradfx(u);
    uy=gradfy(u); 
    normgrad = sqrt(ux.^2+uy.^2+epsilon^2);
    
    v=alpha*u+beta*imfilter(u,H,'circular')-raw;
    
    vf=imfilter(v,H,'circular');
    
    u = u - tau*(vf-lambda*divergence(ux./normgrad,uy./normgrad));

end

init_u=u;

figure,imshow(init_u(5:end-5,5:end-5),[]), title('gd Poisson TV prior'); imcontrast

% keyboard


%% NLM
mex -largeArrayDims ./weights_NLM_adaptive_h.c
w=weights_NLM_adaptive_h(init_u,2,15);


wu=w*reshape(init_u, M*N, 1);
norm_wn=wu./sum(w,2);
NLM=reshape(norm_wn,M,N);

figure, imshow(NLM,[]), title('NLM');


%% NLMpior with Poisson Moded and Iterative Gradient Descent solver
tic
y = raw;
gd_lambda = 0.005;

tau = 0.002;
u = init_u;

w1=sum(w,2);

for iter = 1:200
    
    u1d=reshape(u, M*N, 1);
    NLMprior=u1d.*w1-w*u1d;
    
    uh=imfilter(u,H,'circular');
    v=alpha*u+beta*uh-y;
    
    gradLH=alpha*(ones(M,N)-y./(alpha*u+beta*uh))+beta./(alpha*u+beta*uh).*imfilter(v,H,'circular');
    
    u = u - tau*(gradLH+gd_lambda*reshape(NLMprior,M,N));
    
end
NLMprior_gd=u;
figure,imshow(NLMprior_gd,[]),title('gd Poisson NLM prior');

toc

%% plot all together for comparison
% (5:end-5,5:end-5) to remove the noisy edges during the reconstruction
figure,
subplot(221),imshow(raw(5:end-5,5:end-5),[]), title('Scatter Image');
subplot(222),imshow(init_u(5:end-5,5:end-5),[]), title('gd Poisson TV prior');
subplot(223),imshow(NLM(5:end-5,5:end-5),[]), title('NLM');
subplot(224),imshow(NLMprior_gd(5:end-5,5:end-5),[]),title('gd Poisson NLM prior');







