#include "dataManipulator.h"
#define IGNORED_COLS 3

bool eps_equal(float x, float eps){
    return x > -eps && x < eps;
}

void standardize(csv::CSVReader* reader,std::vector<float>** cols,size_t ncols,size_t nrows){
    
    int removed = 0;
    
    size_t j=0;
    

    size_t fd = 0;
    for (auto& row: *reader) {
        
        bool skip = false;
        //Sample size 1/10
        //if (j%10!=0){skip = true;}
       
        
        if(!skip){
            //Iterate over columns
            fd++;
            for(int i=IGNORED_COLS; i<ncols;i++){
                
                if(row[i].is_int()){
                    
                    cols[i-IGNORED_COLS]->push_back(row[i].get<float>());
                }   
                //Rollback
                else{
                    
                    for(int d=i-1;d>IGNORED_COLS;d--){
                        
                        if(cols[d-IGNORED_COLS]->size() > 0){
                            
                            cols[d-IGNORED_COLS]->pop_back();
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

    float means[ncols-IGNORED_COLS];
    float sds[ncols-IGNORED_COLS];
    for(int c=0;c<ncols-IGNORED_COLS;c++){
        
        accumulator_set<float, stats<tag::mean,tag::variance> > acc;
            for (size_t i=0;i<cols[c]->size();i++){
                
                acc(cols[c]->at(i));
            }
        
        means[c] = mean(acc);
        sds[c] = sqrt(variance(acc));
        //std::cout <<"Mean: "<< means[c] << std::endl;
        //std::cout <<"Std: "<< sds[c] << std::endl;
    }
    //Standardize x = (x - E(X))/SD(X)
    for(int c=0;c<ncols-IGNORED_COLS;c++){
        for (size_t i =0;i<cols[c]->size();i++){
            cols[c]->at(i) = (cols[c]->at(i) - means[c])/sds[c];
        }
    }
    
    
}

int columnNameToIndex(std::vector<std::string>* colNames,std::string colName){
    int i = 0;
    //std::cout << colName;
    for(auto it = (colNames->begin() + IGNORED_COLS);it!= colNames->end();it++){
        //std::cout << *it << std::endl;
        //std::cout << colName << std::endl;
        
        //std::cout << it->compare(colName)<< std::endl;
        if (it->compare(colName) == 0) return i;
        i++;
    }
    return -1;
}
