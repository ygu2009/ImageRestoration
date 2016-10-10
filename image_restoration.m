clc
clear
close all


%% Ground Truth Image
ref = double(imread('cameraman.tif'));
figure, imshow(ref, [0 255]), title('GT');

%% Blur Kernel
H = fspecial('average',9);
% H=fspecial('Gaussian',11,2);
B = imfilter(ref,H,'circular');

%% Add Gaussian Noise
[M, N]=size(B);
sigma=3;
BN = B+randn(M,N)*sigma;
figure, imshow(BN,[]),title(['Raw PSNR=', num2str(PSNR(BN,ref))]);

%% Deconvolution and Denosing by Total Variation (TVL1) Regularization

% calculate the gradient of 2D image
gradfx = @(x)x([2:end end],:)-x;
gradfy = @(x)x(:,[2:end end])-x;


lambda = 0.005; %regularization parameter for TV term
epsilon = 0.1*1e-2;
tau = 0.5;  %step size

u = ones(M,N);

iter=1;
while iter<1000
    iter=iter+1;
    ux=gradfx(u);
    uy=gradfy(u); 
    normgrad = sqrt(ux.^2+uy.^2+epsilon);
    v=imfilter(u,H,'circular')-BN;
    vf=imfilter(v,H,'circular');
    u = u - tau*(vf-lambda*divergence(ux./normgrad,uy./normgrad));
end

init_u=u;
sigma1=sigma*5;

figure,imshow(init_u,[]), imcontrast


%% NLM
mex -largeArrayDims ./weights_NLM.c
w=weights_NLM(init_u,sigma1,2,10);

wu=w*reshape(init_u, M*N, 1);
norm_wn=wu./sum(w,2);
NLM=reshape(norm_wn,M,N);

figure, imshow(NLM,[0 255]), title(['NLM PSNR=', num2str(PSNR(NLM,ref))]);

%% Deconvolution and Denoising by weighted TVL2 Regularization / NLM prior 
lambda2 = 0.005; %regularization parameter for NLM prior term

tau = 0.05; % step size
u = init_u; % initial input

% initial/1st iteration
Eu(1)=1;
w1=sum(w,2);

u1d=reshape(u, M*N, 1);
wu=w*u1d;

NLMprior=u1d.*w1-w*u1d;

v=imfilter(u,H,'circular')-BN;
vf=imfilter(v,H,'circular');

% Keep track of the function is being minimized
Ju=w*(u1d.^2)-2*wu.*u1d + w1.*(u1d.^2);
Hu=norm(v,'fro')^2/(M*N);

Eu(2)=Hu+lambda2*sum(Ju(:))/(M*N);
iter=2;
while iter<2000 & abs(Eu(iter-1)-Eu(iter)) > 1e-5 & tau>1e-5
    
    iter=iter+1;
    
    u = u - tau*(vf+lambda2*reshape(NLMprior,M,N));
    
    u1d=reshape(u, M*N, 1);
    
    NLMprior=u1d.*w1-w*u1d;
    
    
    v=imfilter(u,H,'circular')-BN;
    vf=imfilter(v,H,'circular');
    
    psnr_iter(iter)=PSNR(u,ref);
    
    wu=w*u1d;
    Ju=w1.*(u1d.^2)-2*wu.*u1d+w*(u1d.^2);
    Hu=norm(v,'fro')^2/(M*N);
    
    Eu(iter)=Hu+lambda2*sum(Ju(:))/(M*N);
    
    % line search
    if Eu(iter-1)-Eu(iter) <1e-3 & Eu(iter-1)-Eu(iter) >0
        tau=0.95*tau;
    end
%     if abs(Eu(iter-1)-Eu(iter)) <1e-4
%         lambda2=lambda2*0.95;
%     end
  
    
end

figure,imshow(u,[]);title(['GD NLMprior PSNR=', num2str(PSNR(u,ref))]);
figure, plot(psnr_iter(2:end), 'r'), ylim([0 30]), title('GDNLMprior PSNR'), ylabel('PSNR'), xlabel('iteration')
figure, plot(Eu(2:end),'b'),title('NLMprior Cost Funtion'), ylabel('energy'), xlabel('iteration')
