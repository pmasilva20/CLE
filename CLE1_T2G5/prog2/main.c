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

#include<math.h>
#include <time.h>


double calculateMatrixDeterminant(int size,double matrix[size][size]){
    double determinant=matrix[0][0];

    for (int x = 1; x < size; x++){
        determinant*= matrix[x][x];
    }

    return determinant;

}

//TODO: Verificar se função é da forma que o professor pretende (apesar de funcionar).
double gaussianElimination(int orderMatrix,double matrix[orderMatrix][orderMatrix]){
    //Função para transformar a generic square matrix of order n into an equivalent upper triangular matrix
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

/** \brief worker life cycle routine */
static void *worker (void *id);


/** \brief consumer threads return status array */
int *statusWorks;

int matrixProcessed = 0;

int matrixToProcess =0;

/** File ID*/
int fileid=0;


int main(int argc, char** argv) {


    /**Number of Workers**/
    int numberWorkers=0;

    /** List of Files**/
    char listFiles[10];


    int opt; /* selected option */
    char *fName = "no name"; /* file name (initialized to "no name" by default) */
    opterr = 0;

    do {
        switch ((opt = getopt(argc, argv, ":f:t:h"))) {
            case 'f':
                if (optarg[0] == "-") {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                fName = optarg;
                break;
            case 't':
                numberWorkers = atoi(optarg);
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

    statusWorks = malloc(sizeof(int)*numberWorkers);

    pthread_t tIdWorkers[numberWorkers];

    unsigned int works[numberWorkers];

    int *status_p;

    for (int i = 0; i < numberWorkers; i++)
        works[i] = i;

    srandom ((unsigned int) getpid ());




    //TODO: Inicializar Workers
    for (int i = 0; i < numberWorkers; i++) {
        if (pthread_create(&tIdWorkers[i], NULL, worker, &works[i]) !=0)                             /* thread consumer */
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
        else{
            printf("Thread Worker Created %d !\n", i);
        }
    }




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

    matrixToProcess+=numberMatrices;

    putFileInfo(file_info);
    printf("Main : File %u (%s) to Shared Region.\n",file_info.id,file_info.name);

    for (int i = 0; i < numberMatrices; i++) {

        double matrixDeterminant;

        struct Matrix matrix1;
        matrix1.fileid=fileid;
        matrix1.id=i;
        matrix1.orderMatrix=orderMatrices;

        fread(&matrix1.matrix, sizeof(double), orderMatrices*orderMatrices, pFile);

        putMatrixVal(matrix1);
        printf("Main : Matrix %u to Shared Region.\n",i);
        //gaussianElimination(orderMatrices,matrix1.matrix);
        //matrixDeterminant=calculateMatrixDeterminant(orderMatrices,matrix1.matrix);

        //printf("The determinant is %.3e\n",matrixDeterminant);

    };

    fileid++;

    // TODO: Worker a trabalhar

    // TODO: Imprimir os resultados após terminação dos Workers


    for (int i = 0; i < numberWorkers; i++)
    { if (pthread_join (tIdWorkers[i], (void *) &status_p) != 0)
        {
            fprintf(stderr, "Worker %u : error on waiting for thread producer ",i);
            perror("");
            exit (EXIT_FAILURE);
        }
        printf ("Worker %u : has terminated with status: %d \n", i, *status_p);
    }

    getResults();


    /*int error_code = processMatricesFile(fName);

    if (error_code != 0) {
        printf("Error during file processing of %s", fName);

    }*/

}

static void *worker (void *par)
{
    unsigned int id = *((unsigned int *) par);                                                          /* consumer id */
    struct Matrix val;                                                                                /* produced value */
    int i;/* counting variable */

    do{
        double matrixDeterminant;

        val = getMatrixVal (id);

        printf("Worker %u : Obtained Matrix.\n",id);

        gaussianElimination(val.orderMatrix,val.matrix);

        matrixDeterminant=calculateMatrixDeterminant(val.orderMatrix,val.matrix);

        struct Matrix_result matrix_determinant_result;

        matrix_determinant_result.fileid=val.fileid;

        matrix_determinant_result.id=val.id;

        matrix_determinant_result.determinant=matrixDeterminant;

        printf("Worker %u : Determinant Calculated.\n",id);

        putResults(matrix_determinant_result,id);

        printf("Worker %u : Saved Results obtained.\n",id);
        //printf("Worker %u : Matrix Processed: %u\n",id,matrixProcessed);
    } while (matrixProcessed<matrixToProcess);

    statusWorks[id] = EXIT_SUCCESS;
    pthread_exit (&statusWorks[id]);
}

static void printUsage (char *cmdName)
{
    fprintf (stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                     " OPTIONS:\n"
                     " -h --- print this help\n"
                     " -f --- filename\n"
                     , cmdName);
}
