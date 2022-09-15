#ifndef KMEANSCUDA_H
#define KMEANSCUDA_H
#include <chrono>
#include <iostream>
#include "point.h"
#include <cuda_runtime_api.h>

using namespace std::chrono;


void kmeans_SOA_cuda(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);
__device__
double distanceCuda(double x0,double y0, double x1, double y1);
#endif