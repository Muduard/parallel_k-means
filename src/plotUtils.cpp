#include "plotUtils.h"
void getVecsFromPVec(pVec* p, std::valarray<double>* vecX, std::valarray<double>* vecY){
    std::vector<double> tmpx;
    std::vector<double> tmpy;
    for (auto it = p->begin();it!= p->end();it++){
        tmpx.push_back(it->getX());
        
        tmpy.push_back(it->getY());
    }
    
    *vecX = Vec(tmpx.data(),tmpx.size());
    *vecY = Vec(tmpy.data(),tmpy.size());
}

void cleanCache(){
    double a[2000];
    for (int i=0;i<2000;i++) {
        a[i] = 5;
    }
}

void plotResults(pVec dataset, int k, int epochs){
    Plot plot;
    plot.legend().show(false);
    
    Vec x,y,cx,cy;
    getVecsFromPVec(&dataset,&x,&y);
    
    double bounds[] = {x.min(),x.max(),y.min(),y.max()};
    pVec centroids = randomCentroids(k,bounds);
    getVecsFromPVec(&centroids,&cx,&cy);
    
    
    auto start = high_resolution_clock::now();
    kmeans(&dataset,k,&centroids,epochs,bounds);
    auto stop = high_resolution_clock::now();
    std::cout << "KMEANS: " << duration_cast<milliseconds>(stop-start).count() << " ms"<< std::endl;

    cleanCache();

    start = high_resolution_clock::now();
    parallelKmeans(&dataset,k,&centroids,epochs,bounds);
    stop = high_resolution_clock::now();
    std::cout << "Parallel KMEANS: " << duration_cast<milliseconds>(stop-start).count() << " ms"<< std::endl;

    plot.drawDots(x,y);
    plot.drawDots(cx,cy).lineWidth(5);
    plot.show(); 

    Plot plot2;
    plot2.legend().show(false);
    pVec clusters[k];
    for(auto p = dataset.begin();p!=dataset.end();p++){
        clusters[p->getCluster()].push_back(*p);
        
    }
    Vec clusterDraw[k*2];
    for(int i=0;i<k*2;i+=2){
        
        getVecsFromPVec(&clusters[i/2],&clusterDraw[i],&clusterDraw[i+1]);
        plot2.drawDots(clusterDraw[i],clusterDraw[i+1]);
    }
    Vec cx2,cy2;
    getVecsFromPVec(&centroids,&cx2,&cy2);
    //printPVec(&centroids);
    plot2.drawDots(cx2,cy2).lineWidth(5);
    plot2.show();
}

void plotSpeedUp(std::vector<double>* sequential, std::vector<double>* parallel, Vec* nProcessors){
    std::vector<double> Sp;
    for(int i=0;i<nProcessors->size();i++){
        double avgSeq,avgPar = 0;
        for(int j=0;j<sequential->size();j++){
            avgSeq += sequential->at(j);
            avgPar += parallel->at(i*nProcessors->size() +j);
        }
        avgSeq /= sequential->size();
        avgPar /= sequential->size();
        Sp.push_back(avgSeq/avgPar);
    }
    Vec SpVec(Sp.data(),Sp.size());

    Plot plot;
    plot.drawCurve(*nProcessors,SpVec).label("Speedup");
    plot.legend().atOutsideTopRight();
    plot.xlabel("Processors");
    plot.ylabel("Speedup");
    
    Figure fig = {{ plot }};
    fig.size(600,300);
    fig.show();
}

void allocSOA(pVec* AOS,double** SOA){
    
    SOA[0] = (double*) malloc(AOS->size()*sizeof(double));
    SOA[1] = (double*) malloc(AOS->size()*sizeof(double));
    int j=0;
    
    for(auto p = AOS->begin();p!= AOS->end();p++){
        SOA[0][j] = p->getX();
        SOA[1][j] = p->getY();
        j++;
    }
    
}