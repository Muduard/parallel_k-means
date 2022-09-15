#ifndef POINT_H
#define POINT_H
#include <cmath> 

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
double distance(double x0,double y0, double x1, double y1);
#endif