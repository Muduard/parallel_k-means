
#include <vector>
#include "csv.hpp"
#include "dataManipulator.h"
#include "kmeans.h"
#include <iostream>
#define IGNORED_COLS 3

int main(){

    csv::CSVReader reader("Mall_Customers.csv");
    size_t ncols = csv::get_file_info("Mall_Customers.csv").n_cols;
    size_t nrows = csv::get_file_info("Mall_Customers.csv").n_rows;
    std::cout << "Cols: " << ncols << ", Rows: " << nrows << std::endl;
    

    std::vector<float>* cols[ncols-IGNORED_COLS];
    for(int i =0;i<ncols-IGNORED_COLS;i++){
        cols[i] = new std::vector<float>;
       
    }
    auto start = high_resolution_clock::now();
    standardize(&reader,cols,ncols,nrows);
    auto stop = high_resolution_clock::now();
    std::cout << duration_cast<milliseconds>(stop-start).count() << std::endl;
    std::cout << cols[0]->at(0) << std::endl;
    std::cout << cols[1]->at(0) << std::endl;
    //for(size_t i = 0;i<nrows;i+=10){
    Point p(3.2,2.1);
    Point p2(10.9,2.1);
    std::cout << p.distance(p2) << std::endl;
    //}
    
}