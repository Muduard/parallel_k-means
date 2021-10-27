#ifndef KMEANS_H
#define KMEANS_H

void kmeans(float** dataset,int k, float** result){

}

class Point{
    private:
        double x,y;
        int cluster;
        double minDist;
    public:
        Point();
        Point(double,double);
        double distance(Point p);
} point;

#endif