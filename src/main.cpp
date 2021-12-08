
#include <vector>
#include <string>
#include "csv.hpp"
#include "dataManipulator.h"
#include "kmeans.h"
#include <iostream>
#include <fstream>
#define IGNORED_COLS 3
#include <sciplot/sciplot.hpp>
#include <boost/program_options.hpp>
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include "fmt/format.h"
#ifdef _OPENMP
#include <omp.h> // for OpenMP library functions
#endif
namespace po = boost::program_options;
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

pVec getDataset0(){
    //Read dataset
    csv::CSVReader reader("Mall_Customers.csv");

    //Get number of columns and rows of dataset
    size_t ncols = csv::get_file_info("Mall_Customers.csv").n_cols;
    size_t nrows = csv::get_file_info("Mall_Customers.csv").n_rows;
    
    //Get dataset's column names
    std::vector<std::string> colNames = csv::get_col_names("Mall_Customers.csv");

    //Allocate vectors for columns expect for the ignored ones
    std::vector<float>* cols[ncols-IGNORED_COLS];
    for(int i =0;i<ncols-IGNORED_COLS;i++){
        cols[i] = new std::vector<float>;
    }

    //Standardize data using the formula (x - mean(x))/sd(x)
    auto start = high_resolution_clock::now();
    standardize(&reader,cols,ncols,nrows);
    auto stop = high_resolution_clock::now();
    std::cout << duration_cast<milliseconds>(stop-start).count() << std::endl;
    
    //Allocate vector of points as dataset
    pVec dataset;
    //Define columns of interest for the analysis
    std::string col0 = "Annual Income (k$)";
    std::string col1 = "Spending Score (1-100)";

    //Iterator of col1
    auto it1 = cols[columnNameToIndex(&colNames,col1)]->begin();
    
    //Iteratate on col0 and col1
    for (auto it0 = cols[columnNameToIndex(&colNames,col0)]->begin(); it0 != cols[columnNameToIndex(&colNames,"Annual Income (k$)")]->end();it0++){
        //Add points to datasets
        dataset.push_back(Point(*it0,*it1));
        it1++;
    }
    return dataset;
}

pVec getDataset1(double minX, double maxX, double minY, double maxY, int k,int n){
    std::random_device rd;
    std::default_random_engine eng(rd());
    //Define two uniform distributions in [minX,maxX] and [minY,maxY], one for the x points, one for the y points 
    std::uniform_real_distribution<double> xDistr(minX, maxX);
    std::uniform_real_distribution<double> yDistr(minY, maxY);

    //Allocate vector of points as dataset
    pVec dataset;

    //Generate dataset as a staircase to make easily visible clusters
    for (int j=0;j<k;j++){
        for(int i=0;i<n/k;i++){
            dataset.push_back(Point(xDistr(eng)-j*maxX*2,yDistr(eng)-j*maxY*2));
        }
        
    }
    return dataset;
    
}

void writeVectorToFile(std::vector<double> v, char* filename){
    
    std::ofstream outFile(filename);
    // the important part
    for (const auto &e : v) outFile << e << "\n";
}

void writeVectorToFile(std::vector<double> v, std::string filename){
    std::ofstream outFile(filename);
    // the important part
    for (const auto &e : v) outFile << e << "\n";
}

void plotSpeedUp(std::vector<double>* sequential, std::vector<double>* parallel, Vec nProcessors){
    std::vector<double> Sp;
    for(int i=0;i<sequential->size();i++){
        Sp.push_back(sequential->at(i)/parallel->at(i));
    }
    Vec SpVec(Sp.data(),Sp.size());
    Plot plot;
    plot.drawCurve(nProcessors,SpVec);
    plot.legend().atOutsideTopRight();
    plot.xlabel("Processors");
    plot.ylabel("Speedup");
    
    Figure fig = {{ plot }};
    fig.size(600,300);
    fig.show();
}

void allocSOA(pVec* AOS,double** SOA){
    SOA = (double**) malloc(2*sizeof(double*));
                SOA[0] = (double*) malloc(AOS->size()*sizeof(double));
                SOA[1] = (double*) malloc(AOS->size()*sizeof(double));
                int j=0;
                for(auto p = AOS->begin();p!= AOS->end();p++){
                    SOA[0][j] = p->getX();
                    SOA[1][j] = p->getY();
                    j++;
                }
}

void procTest(int k, int points, int epochs,bool soa,int maxProcNumber,int datapoints){
    std::vector<double> timesParallelProcs[maxProcNumber];
    for(int nProc = 1;nProc <= maxProcNumber;nProc++){
        omp_set_num_threads(nProc);
        std::vector<double> timesVecParallel;
    
        Vec nPoints =  linspace(0, points, datapoints);
        int increment = points/datapoints;
        
        for (int i=increment;i<points;i+=increment){ 
            std::cout << "points: " << i << std::endl;
            pVec dataset = getDataset1(-20000,20000,-20000,20000,k,i);
            Vec x,y,cx,cy;
            getVecsFromPVec(&dataset,&x,&y);

            double bounds[] = {x.min(),x.max(),y.min(),y.max()};

            pVec centroids = randomCentroids(k,bounds);
            getVecsFromPVec(&centroids,&cx,&cy);
            double** datasetSOA;
            double** centroidsSOA;
            if(soa){
                allocSOA(&dataset,datasetSOA);
                allocSOA(&centroids,centroidsSOA);
            }
            
            auto start = high_resolution_clock::now();
            if(soa){
                parallelKmeans_SOA(datasetSOA,dataset.size(),k,centroidsSOA,centroids.size(),epochs,bounds);
            }else{
                parallelKmeans(&dataset,k,&centroids,epochs,bounds);
            }
            auto stop = high_resolution_clock::now();
            timesVecParallel.push_back(duration_cast<seconds>(stop-start).count());
        }
        writeVectorToFile(timesVecParallel,fmt::format("Parallel_{}.txt.", nProc));
        
    }
}


void testPlot(int k, int points, int epochs,bool soa,int datapoints){

    std::vector<double> timesVec,timesVecParallel;
    
    Vec nPoints =  linspace(0, points, datapoints);
    int increment = points/datapoints;
        
    for (int i=increment;i<points;i+=increment){ 
        std::cout << "points: " << i << std::endl;
        pVec dataset = getDataset1(-20000,20000,-20000,20000,k,i);
        Vec x,y,cx,cy;
        getVecsFromPVec(&dataset,&x,&y);

        double bounds[] = {x.min(),x.max(),y.min(),y.max()};

        pVec centroids = randomCentroids(k,bounds);
        getVecsFromPVec(&centroids,&cx,&cy);
        double** datasetSOA;
        double** centroidsSOA;
        if(soa){
            allocSOA(&dataset,datasetSOA);
            allocSOA(&centroids,centroidsSOA);
        }
        
        auto start = high_resolution_clock::now();
        if(soa){
            kmeans_SOA(datasetSOA,dataset.size(),k,centroidsSOA,centroids.size(),epochs,bounds);

        }else{
            kmeans(&dataset,k,&centroids,epochs,bounds);
        }
        
        auto stop = high_resolution_clock::now();
        timesVec.push_back(duration_cast<seconds>(stop-start).count());

        cleanCache();
        
        start = high_resolution_clock::now();
        if(soa){
            parallelKmeans_SOA(datasetSOA,dataset.size(),k,centroidsSOA,centroids.size(),epochs,bounds);
        }else{
            parallelKmeans(&dataset,k,&centroids,epochs,bounds);
        }
       
        stop = high_resolution_clock::now();
        timesVecParallel.push_back(duration_cast<seconds>(stop-start).count());
    }
    writeVectorToFile(timesVec,"times.txt");
    writeVectorToFile(timesVecParallel,"timesParallel.txt");
    Vec times(timesVec.data(),timesVec.size());
    Vec timeParallel(timesVecParallel.data(),timesVecParallel.size());
    Plot plot;
    plot.drawCurve(nPoints,times).label("Normal");
    plot.drawCurve(nPoints,timeParallel).label("Parallel");
    plot.legend().atOutsideTopRight();
    plot.xlabel("Points");
    plot.ylabel("Seconds");
    
    Figure fig = {{ plot }};
    fig.size(600,300);
    fig.show();
    
}

int main(int ac, char* av[]){
    int clusters = 20;
    int epochs = 50;
    int points = 2000;
    bool useDataset = false;
    bool test = false;
    int datapoints = 10;
    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("clusters", po::value<int>(), "set number of clusters")
            ("epochs", po::value<int>(), "set number of epochs")
            ("points", po::value<int>(), "set number of points to be generated for the dataset")
            ("datapoints", po::value<int>(), "set number of data points on which to evaluate the dataset")
            ("use-dataset", po::value<bool>(), "true to use dataset, false to generate dataset")
            ("test", po::value<bool>(), "true to execute test")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }

        if (vm.count("clusters")) {
            clusters = vm["clusters"].as<int>();
        }
        if (vm.count("epochs")) {
            epochs = vm["epochs"].as<int>();
        }
        if (vm.count("points")) {
            points = vm["points"].as<int>();
        }
        if (vm.count("datapoints")) {
            datapoints = vm["datapoints"].as<int>();
        }
        if (vm.count("use-dataset")) {
            useDataset = vm["use-dataset"].as<bool>();
        }
        if (vm.count("test")) {
            test = vm["test"].as<bool>();
        }

    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }
    pVec dataset;
    if(useDataset){
        dataset = getDataset0();
    }else{
        if(!test){
            dataset = getDataset1(-2,2,-2,2,clusters,points);
        }else{
            procTest(clusters, points, epochs,true,8,datapoints);

        }
    }
    if(!test){
       plotResults(dataset,clusters,epochs);
    }
    return 0;

    
}

