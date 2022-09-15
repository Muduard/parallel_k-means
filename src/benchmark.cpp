#include "kmeans.h"
#include "kmeansCuda.h"
#include "ioUtils.h"
#include "plotUtils.h"

#include <boost/program_options.hpp>
#ifdef _OPENMP
#include <omp.h> // for OpenMP library functions
#endif

#define NPROC 8

namespace po = boost::program_options;
std::string resultPath ="./results/";

void procTest(int k, int points, int epochs,bool soa,int maxProcNumber,int datapoints){
    std::vector<double> timesP;
    for(int nProc = 1;nProc <= maxProcNumber;nProc++){
        std::cout << "Procs: " << nProc << std::endl;
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
            double** datasetSOA = (double**) malloc(2*sizeof(double*));
            double** centroidsSOA = (double**) malloc(2*sizeof(double*));

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
            timesVecParallel.push_back(duration_cast<milliseconds>(stop-start).count());
        }
        writeVectorToFile(timesVecParallel,fmt::format("{}Parallel_{}.txt",resultPath, nProc));
        
    }
}

void seqTest(int k, int points, int epochs,bool soa,int datapoints){
    
        std::vector<double> timesVec;
    
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
            double** datasetSOA = (double**) malloc(2*sizeof(double*));
            double** centroidsSOA = (double**) malloc(2*sizeof(double*));
           
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
            timesVec.push_back(duration_cast<milliseconds>(stop-start).count());
        }
        writeVectorToFile(timesVec,fmt::format("{}Sequential.txt",resultPath));
}
void cudaTest(int k, int points, int epochs,int datapoints){
    
        std::vector<double> timesVec;
    
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
            double** datasetSOA = (double**) malloc(2*sizeof(double*));
            double** centroidsSOA = (double**) malloc(2*sizeof(double*));
           
            
            allocSOA(&dataset,datasetSOA);
            allocSOA(&centroids,centroidsSOA);
            
            
            auto start = high_resolution_clock::now();
            kmeans_SOA_cuda(datasetSOA,dataset.size(),k,centroidsSOA,centroids.size(),epochs,bounds);
            auto stop = high_resolution_clock::now();
            timesVec.push_back(duration_cast<milliseconds>(stop-start).count());
        }
        writeVectorToFile(timesVec,fmt::format("{}Cuda.txt",resultPath));

}



int main(int ac, char* av[]){
    int clusters = 20;
    int epochs = 50;
    int points = 500000;
    int datapoints = 10;
    bool soa = false;
    bool cuda = true;
    bool plot = true;
    bool onlyplot = false;
    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("clusters", po::value<int>(), "set number of clusters")
            ("epochs", po::value<int>(), "set number of epochs")
            ("points", po::value<int>(), "set number of points to be generated for the dataset")
            ("datapoints", po::value<int>(), "set number of data points on which to evaluate the dataset")
            ("soa", po::value<bool>(), "true to use Structures of Arrays")
            ("cuda", po::value<bool>(), "true to use cuda")
            ("result-path", po::value<std::string>(), "set path where to save benchmark results")
            ("plot", po::value<bool>(), "true to plot results after test")
            ("onlyplot", po::value<bool>(), "true to only plot results without new tests")
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
        if (vm.count("result-path")) {
            resultPath = vm["result-path"].as<std::string>();
        }
        if (vm.count("soa")) {
            soa = vm["soa"].as<bool>();
        }
        if (vm.count("cuda")) {
            cuda = vm["cuda"].as<bool>();
        }
        if (vm.count("plot")) {
            plot = vm["plot"].as<bool>();
        }
        if (vm.count("onlyplot")) {
            onlyplot = vm["onlyplot"].as<bool>();
        }

    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

    makeResultDir(resultPath);

    pVec dataset;
    if(!onlyplot){
        if(cuda){
            cudaTest(clusters, points, epochs,datapoints);
        }
        seqTest(clusters, points, epochs,soa,datapoints);
        procTest(clusters, points, epochs,soa,NPROC,datapoints);
    }

    if(plot){
        std::vector<double> parallel;
        std::vector<double> sequential;
        std::vector<double> cuda;

        readSpeedUp(resultPath,datapoints,&parallel,&sequential, &cuda );
        Vec nProcessors =  linspace(1, 8, 7);
        plotSpeedUpCuda(&sequential,&parallel, &cuda,&nProcessors);
    }

    return 0;

    
}

