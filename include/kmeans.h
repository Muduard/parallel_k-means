#ifndef KMEANS_H
#define KMEANS_H
#include <math.h>
#include <vector>
#include <random>
#include <iostream>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/framework/accumulator_set.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <cmath> 
using namespace boost::accumulators;
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
using pVec = std::vector<Point>;
void kmeans(pVec* dataset,int k, pVec* centroids,int epochs,double* bounds);
void parallelKmeans(pVec* dataset,int k, pVec* centroids,int epochs, double* bounds);
pVec randomCentroids(int k, double* bounds);
void printPVec(pVec* P);
#endif