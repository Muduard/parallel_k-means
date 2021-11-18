
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


int main(){

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
    Plot plot;
    plot.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    
    Vec x,y,cx,cy;
    getVecsFromPVec(&dataset,&x,&y);
    
    //std::cout << cols[columnNameToIndex(&colNames,caa)]->at(0) << std::endl;
    int k = 3;
    pVec centroids = randomCentroids(k,x.min(),x.max(),y.min(),y.max());
    getVecsFromPVec(&centroids,&cx,&cy);
    std::cout << centroids.size()<<std::endl;
    printPVec(&centroids);
    kmeans(&dataset,k,&centroids,20);

    plot.drawDots(x,y);
    plot.drawDots(cx,cy).lineWidth(5);
    plot.show(); 

    Plot plot2;
    plot2.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
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
    printPVec(&centroids);
    plot2.drawDots(cx2,cy2).lineWidth(5);
    plot2.show();
}