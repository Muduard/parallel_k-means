#include "kmeans.h"




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
    

    for(int e = 0;e<epochs;e++){
       
        
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

void kmeans_SOA(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds){
    SOAPoint points;
    points.xs = dataset[0];
    points.ys = dataset[1];
    points.minDists = (double*) malloc(n*sizeof(double));
    points.clusters = (int*) malloc(n*sizeof(int));
    for(int i = 0;i<n;i++){
        points.minDists[i] = __DBL_MAX__;
    }
    double* cx = centroids[0];
    double* cy = centroids[1];
   
    for(int e = 0;e<epochs;e++){
        //std::cout << "epoch: " << e << std::endl;
        //auto start = high_resolution_clock::now();
        int i = 0;
        double d;
        //Assign points to clusters
        for (int ci = 0;ci< nci;ci++){

            for(int pi = 0;pi<n;pi++){
                d = distance(cx[ci],cy[ci],points.xs[pi],points.ys[pi]);
                //The point is near the centroid
                if(points.minDists[pi] > d){
                    //Set assigned cluster and new minimal distance
                    points.clusters[pi] = i;
                    points.minDists[pi] = d;
                }
            }
            i++;
        }
        //auto stop = high_resolution_clock::now();
        //std::cout << "for:" << (duration_cast<milliseconds>(stop-start).count()) << "s" << std::endl;
        //Define accumulators to calculate x,y means of the points assigned to the k centroids
        accumulator_set<double, stats<tag::mean> > accX[nci];
        accumulator_set<double, stats<tag::mean> > accY[nci];

        //Accumulate points on their assigned centroid accumulator
        for(int pi = 0;pi<n;pi++){
            //std::cout << p->getCluster() << std::endl;
            
            accX[points.clusters[pi]](points.xs[pi]);
            accY[points.clusters[pi]](points.ys[pi]);
        }
        
        for (int ci = 0;ci< nci;ci++){
            //std::cout << i << ": (" << mean(accX[i]) << ", " <<  mean(accY[i]) << ")" << std::endl;
            //If the mean is NaN, the centroid has no assigned points to it
            
            if(!isnan(mean(accX[ci])) || !isnan(mean(accY[ci]))){
                //Set new centroid position
                cx[ci] = mean(accX[ci]);
                cy[ci] = mean(accY[ci]);
                
            }else{
                //Randomize a new position for the centroid
                Point c = rndCentroid(bounds);
                cx[ci] = c.getX();
                cy[ci] = c.getY();
            }
            
        }
        
    }
}

void parallelKmeans_SOA(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds){
    SOAPoint points;
    
    points.xs = dataset[0];
    points.ys = dataset[1];
    
    points.minDists = (double*) malloc(n*sizeof(double));
    points.clusters = (int*) malloc(n*sizeof(int));
    for(int i = 0;i<n;i++){
        points.minDists[i] = __DBL_MAX__;
    }
    double* cx = centroids[0];
    double* cy = centroids[1];
    
    for(int e = 0;e<epochs;e++){
        //auto start = high_resolution_clock::now();
        //std::cout << "Epoch: " << e << std::endl;
        double d;
        
        #pragma omp parallel for
        for(int pi = 0;pi<n;pi++){
            int i=0;
            for (int ci = 0;ci< nci;ci++){
                //Distanc of current point to current evaluated cluster
                d = distance(cx[ci],cy[ci],points.xs[pi],points.ys[pi]);
                //The point is near the centroid
                if(points.minDists[pi] > d){
                    //Set assigned cluster and new minimal distance
                    points.clusters[pi] = i;
                    points.minDists[pi] = d;
                }
                i++;
            }
            
        }
        //auto stop = high_resolution_clock::now();
        //std::cout << "Parallel for:" << (duration_cast<milliseconds>(stop-start).count()) << "s" << std::endl;
        //Define accumulators to calculate x,y means of the points assigned to the k centroids
        accumulator_set<double, stats<tag::mean> > accX[nci];
        accumulator_set<double, stats<tag::mean> > accY[nci];

        //Accumulate points on their assigned centroid accumulator
        for(int pi = 0;pi<n;pi++){
            //std::cout << p->getCluster() << std::endl;
            accX[points.clusters[pi]](points.xs[pi]);
            accY[points.clusters[pi]](points.ys[pi]);
        }
        
        for (int ci = 0;ci< nci;ci++){
            //std::cout << i << ": (" << mean(accX[i]) << ", " <<  mean(accY[i]) << ")" << std::endl;
            //If the mean is NaN, the centroid has no assigned points to it
            
            if(!isnan(mean(accX[ci])) || !isnan(mean(accY[ci]))){
                //Set new centroid position
                cx[ci] = mean(accX[ci]);
                cy[ci] = mean(accY[ci]);
            }else{
                //Randomize a new position for the centroid
                Point c = rndCentroid(bounds);
                cx[ci] = c.getX();
                cy[ci] = c.getY();
            }
            
        }
    }
}




void parallelKmeans_SOA_nocycle(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds){
    SOAPoint points;
    points.xs = dataset[0];
    points.ys = dataset[1];
    points.minDists = (double*) malloc(n*sizeof(double));
    points.clusters = (int*) malloc(n*sizeof(int));

    double* cx = centroids[0];
    double* cy = centroids[1];
    
    for(int e = 0;e<epochs;e++){

        double d;
        int i = 0;
        short updateCond;
        #pragma omp parallel for
        for(int pi = 0;pi<n;pi++){
            #pragma omp simd
            for (int ci = 0;ci< nci;ci++){
                d = distance(cx[ci],cy[ci],points.xs[pi],points.ys[pi]);
                //Distance is lesser than minimum distance
                updateCond = ((d - points.minDists[pi]) || 0); 
                
                //If updateCond == 1 => points.clusters[pi] = i
                //Else if updateCond == 0 => points.clusters[pi] invariato
                points.clusters[pi] =  (i*updateCond) || (-points.clusters[pi] *(updateCond-1));
                points.minDists[pi] = (d*updateCond) || (-points.minDists[pi] *(updateCond-1));
                
                i++;
            }
            i=0;
        }
        
        //Define accumulators to calculate x,y means of the points assigned to the k centroids
        accumulator_set<double, stats<tag::mean> > accX[nci];
        accumulator_set<double, stats<tag::mean> > accY[nci];

        //Accumulate points on their assigned centroid accumulator
        for(int pi = 0;pi<n;pi++){
            //std::cout << p->getCluster() << std::endl;
            accX[points.clusters[pi]](points.xs[pi]);
            accY[points.clusters[pi]](points.ys[pi]);
        }
        
        for (int ci = 0;ci< nci;ci++){
            //std::cout << i << ": (" << mean(accX[i]) << ", " <<  mean(accY[i]) << ")" << std::endl;
            //If the mean is NaN, the centroid has no assigned points to it
            
            if(!isnan(mean(accX[ci])) || !isnan(mean(accY[ci]))){
                //Set new centroid position
                cx[ci] = mean(accX[ci]);
                cy[ci] = mean(accY[ci]);

                
            }else{
                //Randomize a new position for the centroid
                Point c = rndCentroid(bounds);
                cx[ci] = c.getX();
                cy[ci] = c.getY();
            }
            
        }
    }
}

void kmeans_SOA_nocycle(double** dataset,int n,int k, double** centroids,int nci,int epochs, double* bounds){
    SOAPoint points;
    points.xs = dataset[0];
    points.ys = dataset[1];
    points.minDists = (double*) malloc(n*sizeof(double));
    points.clusters = (int*) malloc(n*sizeof(int));

    double* cx = centroids[0];
    double* cy = centroids[1];
    
    for(int e = 0;e<epochs;e++){

        double d;
        int i = 0;
        short updateCond;
        
        for(int pi = 0;pi<n;pi++){
            #pragma omp simd
            for (int ci = 0;ci< nci;ci++){
                d = distance(cx[ci],cy[ci],points.xs[pi],points.ys[pi]);
                //Distance is lesser than minimum distance
                updateCond = ((d - points.minDists[pi]) || 0); 
                
                //If updateCond == 1 => points.clusters[pi] = i
                //Else if updateCond == 0 => points.clusters[pi] invariato
                points.clusters[pi] =  (i*updateCond) || (-points.clusters[pi] *(updateCond-1));
                points.minDists[pi] = (d*updateCond) || (-points.minDists[pi] *(updateCond-1));
                
                i++;
            }
            i=0;
        }
        
        //Define accumulators to calculate x,y means of the points assigned to the k centroids
        accumulator_set<double, stats<tag::mean> > accX[nci];
        accumulator_set<double, stats<tag::mean> > accY[nci];

        //Accumulate points on their assigned centroid accumulator
        for(int pi = 0;pi<n;pi++){
            //std::cout << p->getCluster() << std::endl;
            accX[points.clusters[pi]](points.xs[pi]);
            accY[points.clusters[pi]](points.ys[pi]);
        }
        
        for (int ci = 0;ci< nci;ci++){
            //std::cout << i << ": (" << mean(accX[i]) << ", " <<  mean(accY[i]) << ")" << std::endl;
            //If the mean is NaN, the centroid has no assigned points to it
            
            if(!isnan(mean(accX[ci])) || !isnan(mean(accY[ci]))){
                //Set new centroid position
                cx[ci] = mean(accX[ci]);
                cy[ci] = mean(accY[ci]);

                
            }else{
                //Randomize a new position for the centroid
                Point c = rndCentroid(bounds);
                cx[ci] = c.getX();
                cy[ci] = c.getY();
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

