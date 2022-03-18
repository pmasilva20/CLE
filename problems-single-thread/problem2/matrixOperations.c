
#include <stdio.h>
#include<math.h>



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
        double  arrayCoefficients[orderMatrices*orderMatrices];

        double matrix[orderMatrices][orderMatrices];

        fread(arrayCoefficients, sizeof(double), 2, pFile);

        //arrayCoefficients to Matrix format
        arrayToMatrix(orderMatrices,arrayCoefficients,matrix);

        printMatrix(orderMatrices,matrix);

        printf("\n");

    }

    fclose (pFile);

}

double gaussianElimination(int orderMatrix,int matrix[orderMatrix][orderMatrix]){
    //TODO: Função para transformar a generic square matrix of order n into an equivalent upper triangular matrix
    int i,j,k;
    for(i=0;i<orderMatrix-1;i++){
        //Partial Pivoting
        for(k=i+1;k<orderMatrix;k++){
            //If diagonal element(absolute vallue) is smaller than any of the terms below it
            if(fabs(matrix[i][i])<fabs(matrix[k][i])){
                //Swap the rows
                for(j=0;j<orderMatrix;j++){
                    double temp;
                    temp=matrix[i][j];
                    matrix[i][j]=matrix[k][j];
                    matrix[k][j]=temp;
                }
            }
        }
        //Begin Gauss Elimination
        for(k=i+1;k<orderMatrix;k++){
            double  term=matrix[k][i]/ matrix[i][i];
            for(j=0;j<orderMatrix;j++){
                matrix[k][j]=matrix[k][j]-term*matrix[i][j];
            }
        }
    }

}

int calculateMatrixDeterminant(int size,double matrix[size][size]){
    //TODO: Função calcular determinante de matrix upper triangular
}





