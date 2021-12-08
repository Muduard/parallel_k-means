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
#include <cmath> 
using namespace boost::accumulators;
using namespace std::chrono;
class Point{
    protected:
        double x,y;
        //Current cluster
        int cluster;
        //Current minimal distance to cluster
        double minDist;
    public:
        Point();
        Point(double,double);
        double distance(Point p);
        double getX();
        double getY();
        void setCluster(int c);
        int getCluster();
        double getMinDist();
        void setMinDist(double dist);
};

struct SOAPoint{
    double* xs;
    double* ys;
    int* clusters;
    double* minDists;
};


using pVec = std::vector<Point>;
void kmeans(pVec* dataset,int k, pVec* centroids,int epochs,double* bounds);
void parallelKmeans(pVec* dataset,int k, pVec* centroids,int epochs, double* bounds);
pVec randomCentroids(int k, double* bounds);
void printPVec(pVec* P);
double distance(double x0,double y0, double x1, double y1);
void kmeans_SOA(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);
void parallelKmeans_SOA(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);
void parallelKmeans_SOA_nocycle(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds);
#endif