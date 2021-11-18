#include "kmeans.h"
using namespace boost::math;
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

pVec randomCentroids(int k, double minX, double maxX, double minY, double maxY){
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> xDistr(minX, maxX);
    std::uniform_real_distribution<double> yDistr(minY, maxY);
    pVec result;
    for (int i=0;i<k;i++){
        result.push_back(Point(xDistr(eng),yDistr(eng)));
    }
    return result;
}

void printPVec(pVec* P){
    for(auto it = P->begin();it != P->end();it++){
        std::cout << "(" << it->getX() << "," << it->getY() << "),";
    }
    std::cout << std::endl;
}
void kmeans(pVec* dataset,int k, pVec* centroids,int epochs){
    
   
    for(int e = 0;e<epochs;e++){
       
        for(auto p = dataset->begin();p!=dataset->end();p++){
            double dist = 0;
            int i = 0;
            for (auto c = centroids->begin();c!= centroids->end();c++){
            //Il punto appartiene al centroide
                if(dist < p->distance(*c)){
                
                    p->setCluster(i);
                    dist = p->distance(*c);
                }
                i++;
            }
        }

        Point newCentroids[centroids->size()];
        accumulator_set<double, stats<tag::mean> > accX[centroids->size()];
        accumulator_set<double, stats<tag::mean> > accY[centroids->size()];
        for (auto p = dataset->begin();p!=dataset->end();p++){
            accX[p->getCluster()](p->getX());
            accY[p->getCluster()](p->getY());
        }
        for(int i =0;i<centroids->size();i++){
            std::cout << i << ": (" << mean(accX[i]) << ", " <<  mean(accY[i]) << ")" << std::endl;
            if(!isnan(mean(accX[i]))){
                centroids->at(i) = Point(mean(accX[i]),mean(accY[i]));
            }else{
                std::cout << "NAN" << std::endl;
            }
            
        }
    }
        
    
    
}

void Point::setCluster(int c){
    cluster = c;
}

int Point::getCluster(){
    return cluster;
}