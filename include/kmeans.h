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
using namespace boost::accumulators;
class Point{
    protected:
        double x,y;
        int cluster;
        double minDist;
    public:
        Point();
        Point(double,double);
        double distance(Point p);
        double getX();
        double getY();
        void setCluster(int c);
        int getCluster();
};
using pVec = std::vector<Point>;
void kmeans(pVec* dataset,int k, pVec* centroids,int epochs);
pVec randomCentroids(int k, double minX, double maxX, double minY, double maxY);
void printPVec(pVec* P);
#endif