
#include <stdio.h>
int readMatrixFile(char* filename) {

    //Read files

    double *matrixCoefficients;
    int numberMatrices;
    int orderMatrices;


    //int previousCharacter = 0;

    FILE* pFile;
    pFile = fopen(filename,"r");

    if(pFile == NULL){
        printf("Error reading file\n");
        return 1;
    }
    fread(&numberMatrices, sizeof(int), 1, pFile);
    fread(&orderMatrices, sizeof(int), 1, pFile);


    printf("Number of Matrices %d\n",numberMatrices);
    printf("Order of the Matrices %d\n",orderMatrices);


    for (int i = 0; i < 2 - 1; i++) {
        double  matrix[orderMatrices*orderMatrices];
        fread(matrix, sizeof(double), 2, pFile);
        for(int a=0; a<orderMatrices; a++) {
            printf("%f \n", matrix[a]);
        }
    }
    fclose (pFile);

    /*while( (matrix = getc(pFile)) != EOF){


    }*/
    //return 0;
}

