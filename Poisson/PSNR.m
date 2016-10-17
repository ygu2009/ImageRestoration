function psnr=PSNR(P,ref)
% ref: original image
% P: degraded image

[M, N]=size(ref);

maxval=max([max(ref(:)); max(P(:))]);

MSE=sum(sum((ref-P).^2))/(M*N);
psnr=10*log10(maxval^2/MSE);