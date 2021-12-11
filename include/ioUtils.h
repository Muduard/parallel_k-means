#ifndef IOUTILS_H
#define IOUTILS_H
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/types.h>
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include "fmt/format.h"
#include "point.h"
#include "csv.hpp"
#include "dataManipulator.h"
#define IGNORED_COLS 3
using pVec = std::vector<Point>;
pVec getDataset0();
pVec getDataset1(double minX, double maxX, double minY, double maxY, int k,int n);
void writeVectorToFile(std::vector<double> v, std::string filename);
std::vector<std::string> getTxtFileList(const char* path);
void readSpeedUp(const char* path, int dataPoints, std::vector<double>* parallel, std::vector<double>* sequential);
#endif