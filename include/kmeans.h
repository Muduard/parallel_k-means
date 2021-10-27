#ifndef KMEANS_H
#define KMEANS_H
#include <math.h>
void kmeans(float** dataset,int k, float** result);

class Point{
    protected:
        double x,y;
        int cluster;
        double minDist;
    public:
        Point();
        Point(double,double);
        double distance(Point p);
};

#endif