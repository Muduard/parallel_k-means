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

//Generate a random point, in the specified bounds 
//Bounds must be an 1D-array with values [minX,maxX,minY,maxY]
Point rndCentroid(double* bounds){
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> xDistr(bounds[0], bounds[1]);
    std::uniform_real_distribution<double> yDistr(bounds[2], bounds[3]);
    return Point(xDistr(eng),yDistr(eng));
}

//Generate k points in the specified bounds
pVec randomCentroids(int k, double* bounds){
    
    pVec result;
    for (int i=0;i<k;i++){
        result.push_back(rndCentroid(bounds));
    }
    return result;
}

//Prints a vector of points
void printPVec(pVec* P){
    for(auto it = P->begin();it != P->end();it++){
        std::cout << "(" << it->getX() << "," << it->getY() << "),";
    }
    std::cout << std::endl;
}


void parallelKmeans(pVec* dataset,int k, pVec* centroids,int epochs, double* bounds){
    int nanseq = 0;
    for(int e = 0;e<epochs;e++){
        if(nanseq > 20){
            *centroids = randomCentroids(k,bounds);
            nanseq = 0;
        }
        
        #pragma omp parallel for
        for(int j=0;j<dataset->size();j++){
            int i = 0;
               
            for (auto c = centroids->begin();c!= centroids->end();c++){
                //Il punto appartiene al centroide
                Point* p = &(dataset->at(j));
                
                if(p->getMinDist() > c->distance(*p)){
                    
                    p->setCluster(i);
                    p->setMinDist(c->distance(*p));
                    
                }
                i++;
            }
        }
        
        

        accumulator_set<double, stats<tag::mean> > accX[centroids->size()];
        accumulator_set<double, stats<tag::mean> > accY[centroids->size()];
        for (auto p = dataset->begin();p!=dataset->end();p++){
            //std::cout << p->getCluster() << std::endl;
            accX[p->getCluster()](p->getX());
            accY[p->getCluster()](p->getY());
        }
        for(int i =0;i<centroids->size();i++){
            //std::cout << i << ": (" << mean(accX[i]) << ", " <<  mean(accY[i]) << ")" << std::endl;
            
            if(!isnan(mean(accX[i])) || !isnan(mean(accY[i]))){
                centroids->at(i) = Point(mean(accX[i]),mean(accY[i]));
            }else{
                centroids->at(i) = rndCentroid(bounds);
                nanseq++;
            }
            
        }
    }
        
}
/*Implementation of kmeans: 
    -dataset is a vector of points
    -k is the number of clusters
    -centroids is the vector of points where the result will be stored
    -epochs is the number of iterations
    -bounds specifies the bounds of the random distribution generating the random centroids 
        at the first iteration
    
  Steps of the kmeans algorthm:
    -Assign points to the nearest centroid
    -Calculate the mean distances of the points to the relative assigned centroid
    -Set centroid position to previosly calculated mean
    -Repeat steps for epochs times
    */
void kmeans(pVec* dataset,int k, pVec* centroids,int epochs, double* bounds){
   

    for(int e = 0;e<epochs;e++){

        int i = 0;
        //Assign points to clusters
        for (auto c = centroids->begin();c!= centroids->end();c++){
            
            for(auto p = dataset->begin();p!=dataset->end();p++){

                //The point is near the centroid
                if(p->getMinDist() > c->distance(*p)){
                    //Set assigned cluster and new minimal distance
                    p->setCluster(i);
                    p->setMinDist(c->distance(*p));
                }
            }
            i++;
        }

        //Define accumulators to calculate x,y means of the points assigned to the k centroids
        accumulator_set<double, stats<tag::mean> > accX[centroids->size()];
        accumulator_set<double, stats<tag::mean> > accY[centroids->size()];

        //Accumulate points on their assigned centroid accumulator
        for (auto p = dataset->begin();p!=dataset->end();p++){
            //std::cout << p->getCluster() << std::endl;
            accX[p->getCluster()](p->getX());
            accY[p->getCluster()](p->getY());
        }
        
        for(int i =0;i<centroids->size();i++){
            //std::cout << i << ": (" << mean(accX[i]) << ", " <<  mean(accY[i]) << ")" << std::endl;
            //If the mean is NaN, the centroid has no assigned points to it
            if(!isnan(mean(accX[i])) || !isnan(mean(accY[i]))){
                //Set new centroid position
                centroids->at(i) = Point(mean(accX[i]),mean(accY[i]));
            }else{
                //Randomize a new position for the centroid
                centroids->at(i) = rndCentroid(bounds);
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

double Point::getMinDist(){
    return minDist;
}

void Point::setMinDist(double dist){
    minDist = dist;
}