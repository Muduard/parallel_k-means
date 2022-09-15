#include "kmeansCuda.h"
__device__
double distanceCuda(double x0,double y0, double x1, double y1){
    return sqrt(pow(x1 - x0,2) + pow(y1 - y0,2));
}

__global__
void computeClusterPoints(double* centroidx, double* centroidy, int ci, double* xs, double* ys, double* minDists, int* clusters){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double d = distanceCuda(*centroidx,*centroidy,xs[i],ys[i]);
    //The point is near the centroid
    if(minDists[i] > d){
        //Set assigned cluster and new minimal distance
        clusters[i] = ci;
        minDists[i] = d;
    }
}

__global__
void initAcc(double* accX, double* accY){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    accX[i] = 0;
    accY[i] = 0;
}
__global__
void computeCentroids(double* accX, double* accY, int* clusters, double* xs, double* ys, double* cx, double* cy, int n, int nci){
    for(int pi = 0;pi<n;pi++){
        accX[clusters[pi]] += xs[pi];
        accY[clusters[pi]] += ys[pi];
    }

    for (int ci = 0;ci< nci;ci++){
        //Set new centroid position
        cx[ci] = accX[ci]/n;
        cy[ci] = accY[ci]/n;
    }
}


void kmeans_SOA_cuda(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds){
    SOAPoint* points ;
    
    cudaMalloc((void**) &points, n*sizeof(SOAPoint));
    double* xs,*ys,*minDists, *xsHost, *ysHost, *minDistsHost;
    int* clusters, *clustersHost;

    xsHost = (double *) malloc(n*sizeof(double));
    ysHost = (double *) malloc(n*sizeof(double));
    minDistsHost = (double *) malloc(n*sizeof(double));
    clustersHost = (int *) malloc(n*sizeof(int));
    //points->xs = dataset[0];
    //points->ys = dataset[1];
    cudaMalloc((void**) &(xs),n*sizeof(double));
    cudaMalloc((void**) &(ys),n*sizeof(double));
    cudaMemcpy(xs, dataset[0],(n)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(ys, dataset[1],(n)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMalloc((void**) &(minDists),n*sizeof(double));
    cudaMalloc((void**) &(clusters),n*sizeof(int));

    //cudaMalloc((void**)&cudaImage, imageSize);
    double* cx;
    double* cy;
    cudaMalloc((void**) &cx,(n/2)*sizeof(double));
    cudaMalloc((void**) &cy,(n/2)*sizeof(double));
    cudaMemcpy(cx, centroids[0],(n/2)*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(cy, centroids[1],(n/2)*sizeof(double),cudaMemcpyHostToDevice);



    for(int e = 0;e<epochs;e++){
        //Define accumulators to calculate x,y means of the points assigned to the k centroids
        for (int ci = 0;ci< nci;ci++){
            //Distanc of current point to current evaluated cluster
            computeClusterPoints<<<n/256,256>>>(cx,cy,ci,xs,ys,minDists,clusters);
        }

        cudaMemcpy(cx, centroids[0],(n/2)*sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(cy, centroids[1],(n/2)*sizeof(double),cudaMemcpyHostToDevice);

       double *accX, *accY;
       cudaMalloc((void**) &accX,(n)*sizeof(double));
       cudaMalloc((void**) &accY,(n)*sizeof(double));
       initAcc<<<n/128,128>>>(accX,accY);
       computeCentroids<<<1,1>>>(accX, accY,clusters, xs, ys,  cx,  cy, n, nci);

    }
}