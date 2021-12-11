#include "kmeans.h"
#include "ioUtils.h"
#include "plotUtils.h"

#include <boost/program_options.hpp>
#ifdef _OPENMP
#include <omp.h> // for OpenMP library functions
#endif
namespace po = boost::program_options;
std::string resultPath ="./results/";

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
            timesVecParallel.push_back(duration_cast<seconds>(stop-start).count());
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
            timesVec.push_back(duration_cast<seconds>(stop-start).count());
        }
        writeVectorToFile(timesVec,fmt::format("{}Sequential.txt",resultPath));
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
    writeVectorToFile(timesVec,fmt::format("times.txt",resultPath));
    writeVectorToFile(timesVecParallel,fmt::format("timesParallel.txt",resultPath));
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
    bool soa = false;
    bool plot = false;
    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("clusters", po::value<int>(), "set number of clusters")
            ("epochs", po::value<int>(), "set number of epochs")
            ("points", po::value<int>(), "set number of points to be generated for the dataset")
            ("datapoints", po::value<int>(), "set number of data points on which to evaluate the dataset")
            ("soa", po::value<bool>(), "true to use Structures of Arrays")
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
        if (vm.count("result-path")) {
            resultPath = vm["result-path"].as<std::string>();
        }
        if (vm.count("test")) {
            test = vm["test"].as<bool>();
        }
        if (vm.count("soa")) {
            soa = vm["soa"].as<bool>();
        }
        if (vm.count("plot")) {
            soa = vm["plot"].as<bool>();
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
            if(plot){
                std::vector<double> parallel;
                std::vector<double> sequential;
                readSpeedUp("./results",10,&parallel,&sequential);
                Vec nProcessors =  linspace(1, 8, 7);
                std::cout << parallel.size() << std::endl;
                plotSpeedUp(&sequential,&parallel,&nProcessors);
            }else{
                procTest(clusters, points, epochs,soa,8,datapoints);
                seqTest(clusters, points, epochs,soa,datapoints);
            }
            
            
            
        }
    }
    if(!test){
       plotResults(dataset,clusters,epochs);
    }
    return 0;

    
}

