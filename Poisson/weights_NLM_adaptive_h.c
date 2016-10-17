/*   pixel weights from Non-Local Means with adaptive h for Poisson Noise
 *
 *=================================================================
 *Syntax: w=weights_NLM_adaptive_h(image, h, dd, ss)
 *dd: half of similarity window
 *ss: half of search window
 *h: degree of filtering
 *=================================================================
 *
 * MATLAB C/MEX Code Implementation of the Non local filter for 
 * Paper  "A non-local algorithm for image denoising"
 * Author: A. Buades, B. Coll and J.M. Morel
 */
/*Copyright (c) Yingying Gu 2016-02-02*/

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
    int pwin, dd;  /*patch window size*/
    int swin, ss; /*search window size*/
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
     dd=mxGetScalar(prhs[1]);
     ss=mxGetScalar(prhs[2]);
     
     pwin=(dd*2+1)*(dd*2+1);
     swin=(ss*2+1)*(ss*2+1);
 
     percent_sparse =(double)swin/(double)MN;
     nzmax = (int)ceil((double)MN*(double)MN*percent_sparse); /*amount of storage allocated for nonzero matrix elements*/
     plhs[0] = mxCreateSparse(MN,MN,nzmax,0);
     weights= mxGetPr(plhs[0]);
     irs = mxGetIr(plhs[0]);
     jcs = mxGetJc(plhs[0]);
     
     /* Replicate the boundaries of the input image -Pad array with mirror reflections of itself. */
     int newM=M+2*dd;
     int newN=N+2*dd;
     int xx,yy;
     input2=(double *)malloc(newM*newN*sizeof(double));
     for(j=0;j<newN;j++){  
         if(j<dd) xx=dd-j-1;
         if(j>=dd & j<newN-dd) xx=j-dd;
         if(j>=newN-dd) xx=N-(j-(newN-dd)+1);  
         for(i=0;i<newM;i++){
             if(i<dd) yy=dd-i-1;
             if(i>=dd & i<newM-dd) yy=i-dd;
             if(i>=newM-dd) yy=M-(i-(newM-dd)+1);
             input2[j*newM+i]=input[xx*M+yy];
         }
     }
     
    
     double *kernel, sum_kernel;
     kernel = (double *)malloc(pwin*sizeof(double)); 
     int x, y, n=0;
     double a=1;
     for(x=-dd;x<=dd;x++){
         for(y=-dd;y<=dd;y++){
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
     for(j=dd;j<newN-dd;j++){
         for(i=dd;i<newM-dd;i++){       
             jcs[(j-dd)*M+i-dd] = wn;
             
             /*compute the sum of all element in patch*/
             sum_p=0, pn=0;
             for (xp=j-dd;xp<=j+dd;xp++)
                 for (yp=i-dd;yp<=i+dd;yp++){
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
             for (x=MAX(j-ss,dd);x<=MIN(j+ss,N-1+dd);x++){
                 for (y=MAX(i-ss,dd);y<=MIN(i+ss,M-1+dd);y++){
                     /*patch similarity*/
                     n=0;
                     similarity=0;
                     for(xr=-dd;xr<=dd;xr++){
                         for(yr=-dd;yr<=dd;yr++){ 
                             tmp=input2[(j+xr)*newM+(i+yr)]-input2[(x+xr)*newM+y+yr];
                             similarity += tmp*tmp*kernel[n++]/(2*adaptive_h2);
                         }
                     }
                     weights[wn]=exp(-similarity);
                     irs[wn]=(x-dd)*M+y-dd;
                  
                     wn++;
                 }
             }
             
         }
     }
    jcs[MN]=wn;
    free(input2);
  
    
}
