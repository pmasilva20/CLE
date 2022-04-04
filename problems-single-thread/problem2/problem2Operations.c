
#include <stdio.h>
#include<math.h>
#include "time.h"


double arrayToMatrix(int size,double arrayCoefficients[size*size],double matrix[size][size]){
    int i = 0;
    for (int x = 0; x < size; ++x)
    {
        for (int y = 0; y < size; ++y)
        {
            matrix[y][x] = arrayCoefficients[i];
            ++i;
        }
    }
}

void printMatrix(int size, double matrix[size][size]){
    int row, columns;
    for (row=0; row<size; row++)
    {
        for(columns=0; columns<size; columns++)
        {
            printf("%f     ", matrix[row][columns]);
        }
        printf("\n");
    }
}

double gaussianElimination(int orderMatrix,double matrix[orderMatrix][orderMatrix]){
    //Função para transformar a generic square matrix of order n into an equivalent upper triangular matrix
    //TODO: Verificar a função e talvez adicionar verificação se a matrix já é upper triangular
    int i,j,k;
    for(i=0;i<orderMatrix-1;i++){
        //Begin Gauss Elimination
        for(k=i+1;k<orderMatrix;k++){
            double  term=matrix[k][i]/matrix[i][i];
            for(j=0;j<orderMatrix;j++){
                matrix[k][j]=matrix[k][j]-term*matrix[i][j];
            }
        }
    }

}

double calculateMatrixDeterminant(int size,double matrix[size][size]){
    // if all coefficients ai j = 0, for j > i, the procedure comes to an end and the value of the determinant is zero
    // else Gaussian elimination and calculos

    double determinant=matrix[0][0];

    for (int x = 1; x < size; x++){
        determinant*= matrix[x][x];
    }
    return determinant;

}

int processMatricesFile(char* filename) {

    //Read files

    //Time Variables
    double t0,t1,t2;
    t2=0.0;
    /*open the file and read the number and the order of matrices to be processed*/


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


    printf("Number of Matrices to be read  = %d\n",numberMatrices);
    printf("Matrices order = %d\n",orderMatrices);
    printf("\n");

    for (int i = 0; i < numberMatrices; i++) {

        t0=((double) clock()) / CLOCKS_PER_SEC;

        printf("Processing matrix %d\n",i+1);

        double  arrayCoefficients[orderMatrices*orderMatrices];

        double matrixDeterminant;

        double matrix[orderMatrices][orderMatrices];

        fread(&matrix, sizeof(double), orderMatrices*orderMatrices, pFile);

        //arrayCoefficients to Matrix format

        //arrayToMatrix(orderMatrices,arrayCoefficients,matrix);

        gaussianElimination(orderMatrices,matrix);

        matrixDeterminant=calculateMatrixDeterminant(orderMatrices,matrix);

        t1=((double) clock()) / CLOCKS_PER_SEC;

        t2+=t1-t0;

        printf("The determinant is %.3e\n",matrixDeterminant);

    }

    printf("\nElapsed time = %.6f s\n",t2);

    fclose (pFile);



}









