#ifndef KMEANS_H
#define KMEANS_H
#include <math.h>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/framework/accumulator_set.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include "point.h"

using namespace boost::accumulators;
using namespace std::chrono;
using namespace boost::math;
using pVec = std::vector<Point>;
void kmeans(pVec* dataset,int k, pVec* centroids,int epochs,double* bounds);
void parallelKmeans(pVec* dataset,int k, pVec* centroids,int epochs, double* bounds);
void kmeans_nocycle(pVec* dataset,int k, pVec* centroids,int epochs,double* bounds);
void kmeans_nocycle2(pVec* dataset,int k, pVec* centroids,int epochs,double* bounds);
void parallelKmeans_nocycle(pVec* dataset,int k, pVec* centroids,int epochs, double* bounds);
void parallelKmeans_nocycle2(pVec* dataset,int k, pVec* centroids,int epochs, double* bounds);
pVec randomCentroids(int k, double* bounds);
void printPVec(pVec* P);
void kmeans_SOA(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);
void parallelKmeans_SOA(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);
void parallelKmeans_SOA_nocycle(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);
void kmeans_SOA_nocycle(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);

#endif