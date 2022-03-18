#include <stdio.h>
#include "./matrixOperations.h"
int main(int argc, char** argv) {
    for(int textIdx = 1; textIdx < argc; textIdx++){

        int error_code = readMatrixFile(argv[textIdx]);

        if(error_code != 0){
            printf("Error during file processing of %s",argv[textIdx]);
            continue;
        }
    }

}
