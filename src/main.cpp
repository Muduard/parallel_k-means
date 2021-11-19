
#include <vector>
#include <string>
#include "csv.hpp"
#include "dataManipulator.h"
#include "kmeans.h"
#include <iostream>
#define IGNORED_COLS 3
#include <sciplot/sciplot.hpp>
using namespace sciplot;
using pVec = std::vector<Point>;

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

void plotResults(pVec dataset, int k){
    Plot plot;
    plot.legend().show(false);
    
    Vec x,y,cx,cy;
    getVecsFromPVec(&dataset,&x,&y);
    
    double bounds[] = {x.min(),x.max(),y.min(),y.max()};
    pVec centroids = randomCentroids(k,bounds);
    getVecsFromPVec(&centroids,&cx,&cy);
    std::cout << centroids.size()<<std::endl;
    
    auto start = high_resolution_clock::now();
    kmeans(&dataset,k,&centroids,100,bounds);
    auto stop = high_resolution_clock::now();
    std::cout << "KMEANS: " << duration_cast<milliseconds>(stop-start).count() << " ms"<< std::endl;

    cleanCache();

    start = high_resolution_clock::now();
    parallelKmeans(&dataset,k,&centroids,100,bounds);
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

pVec getDataset0(){
    csv::CSVReader reader("Mall_Customers.csv");
    size_t ncols = csv::get_file_info("Mall_Customers.csv").n_cols;
    size_t nrows = csv::get_file_info("Mall_Customers.csv").n_rows;
    
    std::vector<std::string> colNames = csv::get_col_names("Mall_Customers.csv");

    std::vector<float>* cols[ncols-IGNORED_COLS];
    for(int i =0;i<ncols-IGNORED_COLS;i++){
        cols[i] = new std::vector<float>;
       
    }
    auto start = high_resolution_clock::now();
    standardize(&reader,cols,ncols,nrows);
    auto stop = high_resolution_clock::now();
    std::cout << duration_cast<milliseconds>(stop-start).count() << std::endl;
    
    pVec dataset;
    std::string col0 = "Annual Income (k$)";
    std::string col1 = "Spending Score (1-100)";
    auto it1 = cols[columnNameToIndex(&colNames,col1)]->begin();
    
    for (auto it0 = cols[columnNameToIndex(&colNames,col0)]->begin(); it0 != cols[columnNameToIndex(&colNames,"Annual Income (k$)")]->end();it0++){
        dataset.push_back(Point(*it0,*it1));
        it1++;
    }
    return dataset;
}

pVec getDataset1(double minX, double maxX, double minY, double maxY, int k,int n){
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> xDistr(minX, maxX);
    std::uniform_real_distribution<double> yDistr(minY, maxY);
    pVec dataset;
    for (int j=0;j<k;j++){
        for(int i=0;i<n/k;i++){
            dataset.push_back(Point(xDistr(eng)-j*maxX*2,yDistr(eng)-j*maxY*2));
        }
    }
    return dataset;
    
}

int main(){
    int k = 20;
    pVec dataset = getDataset1(-2,2,-2,2,k,2000);
    plotResults(dataset,k);
}

