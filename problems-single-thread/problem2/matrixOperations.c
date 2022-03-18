
#include <stdio.h>
int readMatrixFile(char* filename) {

    //Read files

    double *matrixCoefficients;
    int numberMatrices;
    int orderMatrices;


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

    for (int i = 0; i < numberMatrices - 1; i++) {
        double  matrix[orderMatrices*orderMatrices];
        fread(matrix, sizeof(double), 2, pFile);
        for(int a=0; a<orderMatrices*orderMatrices; a++) {
            printf("%f \n", matrix[a]);
        }
        printf("\n");

    }

    fclose (pFile);

}

int gaussianElimination(double matrix[]){
    //TODO: Função para transformar a generic square matrix of order n into an equivalent upper triangular matrix
}

int calculateMatrixDeterminant(double matrix[]){
    //TODO: Função calcular determinante de matrix upper triangular
}

