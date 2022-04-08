#include <stdio.h>
#include "./problem2Operations.h"
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include "fifo.h"
#include <string.h>
#include "structures.h"
#include <pthread.h>
#include "probConst.h"

//TODO: Não a funcionar externa
double calculateMatrixDeterminant(int size,double matrix[size][size]){
    // if all coefficients ai j = 0, for j > i, the procedure comes to an end and the value of the determinant is zero
    // else Gaussian elimination and calculos

    double determinant=matrix[0][0];

    for (int x = 1; x < size; x++){
        determinant*= matrix[x][x];
    }
    return determinant;

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

static void printUsage(char *cmdName);

/** \brief producer threads return status array */
int statusProd[2];

/** \brief consumer threads return status array */
int statusCons[2];


/** File ID*/
int fileid=0;


int main(int argc, char** argv) {


    int opt; /* selected option */
    char *fName = "no name"; /* file name (initialized to "no name" by default) */
    opterr = 0;
    do {
        switch ((opt = getopt(argc, argv, "f:n:h"))) {
            case 'f': /* file name */
                if (optarg[0] == "-") {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                fName = optarg;
                break;
            case 'h' : /* help mode */
                printUsage(basename(argv[0]));
                return EXIT_SUCCESS;
            case '?': /* invalid option */
                fprintf (stderr, "%s: invalid option\n", basename (argv[0]));
                printUsage (basename (argv[0]));
                return EXIT_FAILURE;
            case -1: break;
        }
    } while (opt != -1);

    /*for(int textIdx = 1; textIdx < argc; textIdx++){

        int error_code = readMatrixFile(argv[textIdx]);

        if(error_code != 0){
            printf("Error during file processing of %s",argv[textIdx]);
            continue;
        }
    }*/
    if (argc == 1){
        fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
        printUsage (basename (argv[0]));
        return EXIT_FAILURE;
    }

    //TODO: Inicializar Workers



    //TODO: Armazenar as Matrices na zona partilhada

    FILE* pFile;
    pFile = fopen(fName,"r");

    if(pFile == NULL){
        printf("Error reading file\n");
        return 1;
    }

    struct File_matrices file_info;

    int numberMatrices;
    int orderMatrices;

    file_info.id=fileid;

    strcpy(file_info.name, fName);

    fread(&numberMatrices, sizeof(int), 1, pFile);
    fread(&orderMatrices, sizeof(int), 1, pFile);

    printf("Number of Matrices to be read  = %d\n",numberMatrices);
    printf("Matrices order = %d\n",orderMatrices);
    printf("\n");

    putFileInfo(file_info);

    for (int i = 0; i < numberMatrices; i++) {
        double matrixDeterminant;

        struct Matrix matrix1;
        matrix1.fileid=fileid;
        matrix1.id=i;
        matrix1.orderMatrix=orderMatrices;



        fread(&matrix1.matrix, sizeof(double), orderMatrices*orderMatrices, pFile);


        gaussianElimination(orderMatrices,matrix1.matrix);
        matrixDeterminant=calculateMatrixDeterminant(orderMatrices,matrix1.matrix);

        printf("The determinant is %.3e\n",matrixDeterminant);

    };




    fileid++;

    // TODO: Worker a trabalhar

    // TODO: Imprimir os resultados após terminação dos Workers



    /*int error_code = processMatricesFile(fName);

    if (error_code != 0) {
        printf("Error during file processing of %s", fName);

    }*/

}



static void printUsage (char *cmdName)
{
    fprintf (stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                     " OPTIONS:\n"
                     " -h --- print this help\n"
                     " -f --- filename\n"
                     , cmdName);
}
