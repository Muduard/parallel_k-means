#include "point.h"
Point::Point(){
    x = 0;
    y = 0;
    cluster = -1;
    minDist = __DBL_MAX__;
}

Point::Point(double x, double y){
    this->x = x;
    this->y = y;
    cluster = -1;
    minDist = __DBL_MAX__;
}


double Point::distance(Point p){
    return sqrt(pow(p.x - x,2) + pow(p.y - y,2));
}

double Point::getX(){
    return x;
}
double Point::getY(){
    return y;
}

double distance(double x0,double y0, double x1, double y1){
    return sqrt(pow(x1 - x0,2) + pow(y1 - y0,2));
}

