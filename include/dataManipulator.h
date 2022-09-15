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
bool eps_equal(float x, float eps);


void standardize(csv::CSVReader* reader,std::vector<float>** cols,size_t ncols,size_t nrows);

int columnNameToIndex(std::vector<std::string> *colNames,std::string colName);


#endif