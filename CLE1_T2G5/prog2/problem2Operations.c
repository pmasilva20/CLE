
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

/*
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
*/










