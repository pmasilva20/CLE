#include <stdio.h>
#include "./matrixOperations.h"


int main(int argc, char** argv) {
    /*
    double disp[3][3] = {
            {1, 2, 3},
            {4, 5, 6},
            {3,2,1}};

    gaussianElimination(3,disp);
    printMatrix(3,disp);
    */

    for(int textIdx = 1; textIdx < argc; textIdx++){

        int error_code = readMatrixFile(argv[textIdx]);

        if(error_code != 0){
            printf("Error during file processing of %s",argv[textIdx]);
            continue;
        }
    }


}
