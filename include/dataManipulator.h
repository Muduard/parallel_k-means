#ifndef DATAMANIPULATOR_H
#define DATAMANIPULATOR_H
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "csv.hpp"
#include <chrono>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/framework/accumulator_set.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <math.h> 
using namespace std::chrono;
using namespace boost::accumulators;

//Equality with machine error
bool eps_equal(float x, float eps){
    return x > -eps && x < eps;
}


void standardize(csv::CSVReader* reader,std::vector<float>** cols,size_t ncols,size_t nrows){
    
    int removed = 0;
    
    size_t j=0;
    

    size_t fd = 0;
    for (auto& row: *reader) {
        
        bool skip = false;
        if (j%10!=0){skip = true;}
       
        
        if(!skip){
            //Iterate over columns
            
            for(int i=2; i<ncols;i++){
                if(row[i].is_float()){
                    cols[i-2]->push_back(row[i].get<float>());
                }   
                //Rollback
                else{
                    
                    for(int d=i-1;d>2;d--){
                        
                        if(cols[d-2]->size() > 0){
                            
                            cols[d-2]->pop_back();
                        }
                        
                    }
                    break;
                }
                    
            }
            
            //std::for_each(cols[0].begin(), cols[0].end(), std::bind<void>(std::ref(acc)));
            //std::cout << mean(acc) << std::endl;
            //std::cout << sqrt(variance(acc)) << std::endl;
        }
        j++;
    }
    std::cout << "J: " << j << std::endl;
    std::cout << "fd: " << fd << std::endl;
    std::cout << "Size:" << cols[0]->size() << std::endl;
    float means[ncols-2];
    float sds[ncols-2];
    for(int c=0;c<ncols-2;c++){
        
        accumulator_set<float, stats<tag::mean,tag::variance> > acc;
            for (size_t i=0;i<cols[c]->size();i++){
                
                acc(cols[c]->at(i));
            }
        
        means[c] = mean(acc);
        sds[c] = sqrt(variance(acc));
        std::cout <<"Mean: "<< means[c] << std::endl;
        std::cout <<"Std: "<< sds[c] << std::endl;
    }
    //Standardize x = (x - E(X))/SD(X)
    for(int c=0;c<ncols-2;c++){
        for (size_t i =0;i<cols[c]->size();i++){
            cols[c]->at(i) = (cols[c]->at(i) - means[c])/sds[c];
        }
    }
    
    
}

template <typename T> void removeInvalidRows(std::string columnName,csv::CSVReader* reader){
    
    int removed = 0;
    for (auto& row: *reader) {
        // Note: Can also use index of column with [] operator
        if(row[columnName].is_float()){
            if(row[columnName].get<T>() == 0){
                std::cout <<"removing" << std::endl;
            }
        }
        
    }
    
    
}

#endif