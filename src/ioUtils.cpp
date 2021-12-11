#include "ioUtils.h"
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


void writeVectorToFile(std::vector<double> v, std::string filename){
    std::ofstream outFile(filename);
    // the important part
    for (const auto &e : v) outFile << e << "\n";
}

std::vector<std::string> getTxtFileList(const char* path){
    std::vector<std::string> fileList;
    struct dirent *entry;
    DIR *dir =opendir(path);
    if(dir == NULL){
        std::vector<std::string> a;
        return a;
    }
    while ((entry = readdir(dir)) != NULL) {
        std::string fileName = entry->d_name;
        if (fileName.find(".txt") != std::string::npos) {
            fileList.push_back(entry->d_name);
        }
    }
    closedir(dir);
    return fileList;
}

void readSpeedUp(const char* path, int dataPoints, std::vector<double>* parallel, std::vector<double>* sequential){
    std::vector<std::string> fileList = getTxtFileList(path);
    int i=0;
    for(auto it=fileList.begin();it!= fileList.end();it++){
        std::ifstream plotDataFile;
	    plotDataFile.open(fmt::format("{}/{}",path,*it), std::ios::in);
        if (!plotDataFile) {
            std::cout << "No such file"<<std::endl;
        }
        else {
            std::string cursor;
            
            if(it->find("Sequential") != std::string::npos){
                
                while(std::getline(plotDataFile, cursor)){
                    sequential->push_back(std::atof(cursor.c_str()));
                    i++;
                }
            }else{
                
                while(std::getline(plotDataFile, cursor)){
                    parallel->push_back(std::atof(cursor.c_str()));
                    i++;
                }
            }
            
        }
        plotDataFile.close();
    }
}