#ifndef PLOTUTILS_H
#define PLOTUTILS_H
#include <vector>
#include <iostream>
#include <valarray>
#include <sciplot/sciplot.hpp>
#include "kmeans.h"
using namespace sciplot;
using pVec = std::vector<Point>;

void getVecsFromPVec(pVec* p, std::valarray<double>* vecX, std::valarray<double>* vecY);
void cleanCache();
void plotResults(pVec dataset, int k, int epochs);
void plotSpeedUp(std::vector<double>* sequential, std::vector<double>* parallel, Vec* nProcessors);
void allocSOA(pVec* AOS,double** SOA);
#endif