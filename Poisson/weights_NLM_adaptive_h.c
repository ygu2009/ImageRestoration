/*   
 *   Author: Yingying Gu
 *   Version 1.0
 *   Copyright 2016
 *   University of Wisconsin-Milwaukee
 *
 *   pixel weights from Non-Local Means for Poisson Noise
 *   =================================================================
 *   Syntax: w=weights_NLM_adaptive_h(image, hfpwin, hfswin)
 *   hfpwin: half of similarity patch window
 *   hfswin: half of search window
 *   h: degree of filtering
 *   =================================================================
 *
 *   For Possion noise model, this is modified Implementation of the 
 *   Non local filter for Paper  "A non-local algorithm for image denoising"
 *   Author: A. Buades, B. Coll and J.M. Morel
 */


#include "mex.h" /* Always include this */
#include <math.h>


#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))


void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
                int nrhs, const mxArray *prhs[]) /* Input variables */
{
    double *input; /*input image*/
    double *weights; /*output weights*/
    double *input2; /*add padding to input image*/
    double h;
    int M,N,MN;
    int pwin, hfpwin;  /*patch window size*/
    int swin, hfswin; /*search window size*/
    int i,j,k;
    double tmp, similarity;
    
    double percent_sparse;
    int nzmax;  /*for sparse matrix*/
    mwIndex *irs,*jcs;
   
    
    /*Get dimensions of input image*/
    M=mxGetM(prhs[0]);
    N=mxGetN(prhs[0]);
    input=mxGetPr(prhs[0]); /*Get the pointer to the input image*/
    MN=M*N;
     
     /*Get patch window size and searching window size*/
     hfpwin=mxGetScalar(prhs[1]);
     hfswin=mxGetScalar(prhs[2]);
     
     pwin=(hfpwin*2+1)*(hfpwin*2+1);
     swin=(hfswin*2+1)*(hfswin*2+1);
 
     percent_sparse =(double)swin/(double)MN;
     nzmax = (int)ceil((double)MN*(double)MN*percent_sparse); /*amount of storage allocated for nonzero matrix elements*/
     plhs[0] = mxCreateSparse(MN,MN,nzmax,0);
     weights= mxGetPr(plhs[0]);
     irs = mxGetIr(plhs[0]);
     jcs = mxGetJc(plhs[0]);
     
     /* Replicate the boundaries of the input image -Pad array with mirror reflections of itself. */
     int newM=M+2*hfpwin;
     int newN=N+2*hfpwin;
     int xx,yy;
     input2=(double *)malloc(newM*newN*sizeof(double));
     for(j=0;j<newN;j++){  
         if(j<hfpwin) xx=hfpwin-j-1;
         if(j>=hfpwin & j<newN-hfpwin) xx=j-hfpwin;
         if(j>=newN-hfpwin) xx=N-(j-(newN-hfpwin)+1);  
         for(i=0;i<newM;i++){
             if(i<hfpwin) yy=hfpwin-i-1;
             if(i>=hfpwin & i<newM-hfpwin) yy=i-hfpwin;
             if(i>=newM-hfpwin) yy=M-(i-(newM-hfpwin)+1);
             input2[j*newM+i]=input[xx*M+yy];
         }
     }
     
    
     double *kernel, sum_kernel;
     kernel = (double *)malloc(pwin*sizeof(double)); 
     int x, y, n=0;
     double a=1;
     for(x=-hfpwin;x<=hfpwin;x++){
         for(y=-hfpwin;y<=hfpwin;y++){
             kernel[n] = exp(-(x*x+y*y)/(2.*a*a));
             sum_kernel+=kernel[n];
             /*mexPrintf("kernel[n]=%f ", kernel[n]);*/
             n++;
         }
         /*mexPrintf("\n");*/
     }
     /*normalization of kernel*/
     for(i=0;i<n;i++)
         kernel[i]=kernel[i]/sum_kernel; 
     
     /*weights for every pixel*/
     int xr, yr, wn;
     int xp, yp, pn, t;
     double sum_p, sum_p2, avg, variance, adaptive_h2;
     double *patch;
     patch=(double *)malloc(pwin*sizeof(double)); /*for computing adaptive h in local patch 2016-02-02 ygu*/
     for(j=hfpwin;j<newN-hfpwin;j++){
         for(i=hfpwin;i<newM-hfpwin;i++){       
             jcs[(j-hfpwin)*M+i-hfpwin] = wn;
             
             /*compute the sum of all element in patch*/
             sum_p=0, pn=0;
             for (xp=j-hfpwin;xp<=j+hfpwin;xp++)
                 for (yp=i-hfpwin;yp<=i+hfpwin;yp++){
                     patch[pn]=input2[xp*newM+yp];
                     sum_p += patch[pn];
                     pn++;
                 }
             avg=sum_p/(double)pwin;
             /*compute the variance and standard variation for h  */
             sum_p2=0;
             for (t=pwin;t--;) sum_p2 = sum_p2 + pow((patch[t]-avg),2);
             variance = sum_p2/(double)pwin;
             adaptive_h2=variance;
             /*mexPrintf("adaptive_h2=%f\n", adaptive_h2);*/
             
             /*loop in searching window*/
             for (x=MAX(j-hfswin,hfpwin);x<=MIN(j+hfswin,N-1+hfpwin);x++){
                 for (y=MAX(i-hfswin,hfpwin);y<=MIN(i+hfswin,M-1+hfpwin);y++){
                     /*patch similarity*/
                     n=0;
                     similarity=0;
                     for(xr=-hfpwin;xr<=hfpwin;xr++){
                         for(yr=-hfpwin;yr<=hfpwin;yr++){ 
                             tmp=input2[(j+xr)*newM+(i+yr)]-input2[(x+xr)*newM+y+yr];
                             similarity += tmp*tmp*kernel[n++]/(2*adaptive_h2);
                         }
                     }
                     weights[wn]=exp(-similarity);
                     irs[wn]=(x-hfpwin)*M+y-hfpwin;
                  
                     wn++;
                 }
             }
             
         }
     }
    jcs[MN]=wn;
    free(input2);
  
    
}
