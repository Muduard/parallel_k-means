
#include <vector>
#include "csv.hpp"
#include "dataManipulator.h"
#include <iostream>


int main(){

    csv::CSVReader reader("minute_weather.csv");
    size_t ncols = csv::get_file_info("minute_weather.csv").n_cols;
    size_t nrows = csv::get_file_info("minute_weather.csv").n_rows;
    std::cout << "Cols: " << ncols << ", Rows: " << nrows << std::endl;
    std::string colName0("rain_accumulation");

    std::vector<float>* cols[ncols-2];
    for(int i =0;i<ncols-2;i++){
        cols[i] = new std::vector<float>;
       
    }
    auto start = high_resolution_clock::now();
    standardize(&reader,cols,ncols,nrows);
    auto stop = high_resolution_clock::now();
    std::cout << duration_cast<milliseconds>(stop-start).count() << std::endl;
    std::cout << cols[0]->at(0) << std::endl;
    std::cout << cols[1]->at(0) << std::endl;
    //for(size_t i = 0;i<nrows;i+=10){

    //}
    
}