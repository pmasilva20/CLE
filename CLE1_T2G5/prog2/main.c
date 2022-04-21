#include <stdio.h>
#include "./problem2Operations.h"
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include "fifo.h"
#include <string.h>
#include "structures.h"
#include "probConst.h"
#include <pthread.h>
#include <time.h>
#include <math.h>

/**
 * Calculate Matrix Determinant
 * \param size Order of the Matrix
 * \param matrix Matrix
 * \return Determinant of the Matrix
 */
double calculateMatrixDeterminant(int size,double matrix[size][size]){
    double determinant=matrix[0][0];

    for (int x = 1; x < size; x++){
        determinant*= matrix[x][x];
    }

    return determinant;

}

//TODO: Verificar se função é da forma que o professor pretende (apesar de funcionar).

/**
 * Apply Gaussian Elimination
 * \param orderMatrix Order of the Matrix
 * \param matrix Matrix
 * \return Matrix applied with Gaussian Elimination
 */
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

/** \brief Number of Matrices processed by workers **/
int  matrixProcessed = 0;

/** \brief  Number of Matrices to be processed by workers **/
int matrixToProcess =0;

/** \brief Number of Files Still to be processed **/
int filesStillToProcess =0;

/** \brief Number of Files to be processed **/
int filesToProcess =0;

int main(int argc, char** argv) {

    /** time limits **/
    double t0, t1, t2;

    t2 = 0.0;

    /** \brief Number of Workers **/
    int numberWorkers=0;

    /** \brief  List of Files**/
    char *listFiles[N];

    /** \brief File ID */
    int fileid=0;

    /** selected option */
    int opt;
    opterr = 0;

    /**
     * Command Line Processing
     */
    do {
        switch ((opt = getopt(argc, argv, ":f:t:h"))) {
            case 'f':
                if (optarg[0] == "-") {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                listFiles[fileid]=optarg;
                fileid++;
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

    //TODO: Talvez tirar isto
    if (argc == 1){
        fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
        printUsage (basename (argv[0]));
        return EXIT_FAILURE;
    }


    /** \brief  Files Still To Process **/
    filesStillToProcess=fileid;

    /** \brief Files to Process */
    filesToProcess=fileid;

    statusWorks = malloc(sizeof(int)*numberWorkers);

    /** Workers internal thread id array */
    pthread_t tIdWorkers[numberWorkers];

    /** Workers application defined thread id array*/
    unsigned int works[numberWorkers];

    int *status_p;

    for (int i = 0; i < numberWorkers; i++)
        works[i] = i;

    srandom ((unsigned int) getpid ());



    /**
     * Generation of intervening Workers threads
     */
    for (int i = 0; i < numberWorkers; i++) {
        if (pthread_create(&tIdWorkers[i], NULL, worker, &works[i]) !=0)                             /* thread worker */
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
        else{
            printf("Thread Worker Created %d !\n", i);
        }
    }
    t0 = ((double) clock ()) / CLOCKS_PER_SEC;

    /**
     * File Processing and Matrices to Shared Region
     */
    for (int i = 0; i < filesToProcess; i++) {

        FILE *pFile;
        pFile = fopen(listFiles[i], "r");

        if (pFile == NULL) {
            printf("Error reading File: %s\n",listFiles[i]);
        }
        else {

            struct File_matrices file_info;

            int numberMatrices;
            int orderMatrices;

            file_info.id = i;

            strcpy(file_info.name, listFiles[i]);

            fread(&numberMatrices, sizeof(int), 1, pFile);
            fread(&orderMatrices, sizeof(int), 1, pFile);

            printf("File %u - Number of Matrices to be read  = %d\n", file_info.id, numberMatrices);
            printf("File %u - Matrices order = %d\n", file_info.id, orderMatrices);
            printf("\n");

            file_info.numberOfMatrices=numberMatrices;

            matrixToProcess += numberMatrices;

            //TODO: Verificar onde libertar memória
            file_info.determinant_result = malloc(sizeof(struct Matrix_result) * numberMatrices);
            putFileInfo(file_info);

            printf("Main : File %u (%s) to Shared Region.\n", file_info.id, file_info.name);

            for (int i = 0; i < numberMatrices; i++) {

                struct Matrix matrix1;

                matrix1.fileid = file_info.id;
                matrix1.id = i;
                matrix1.orderMatrix = orderMatrices;

                fread(&matrix1.matrix, sizeof(double), orderMatrices * orderMatrices, pFile);

                putMatrixVal(matrix1);

                printf("Main : Matrix %u to Shared Region.\n", i);

            };
        }
        /** Decrease Number of Files to Process **/
        filesStillToProcess--;
    };

    /** Waiting for the termination of the Workers threads */
    for (int i = 0; i < numberWorkers; i++)
    { if (pthread_join (tIdWorkers[i], (void *) &status_p) != 0)
        {
            fprintf(stderr, "Worker %u : error on waiting for thread producer ",i);
            perror("");
            exit (EXIT_FAILURE);
        }
        printf ("Worker %u : has terminated with status: %d \n", i, *status_p);
    }

    t1 = ((double) clock ()) / CLOCKS_PER_SEC;

    t2 += t1-t0;

    /** Print Final Results  */
    getResults(filesToProcess);

    /** Print Elapsed Time */
    printf ("\nElapsed time = %.6f s\n", t2);
}

/**
 * \brief Function worker.
 * \param par pointer to application defined worker identification
 */
static void *worker (void *par)
{
    /** Worker ID */
    unsigned int id = *((unsigned int *) par);

    /** Matrix Value */
    struct Matrix val;

    while ((matrixProcessed<matrixToProcess) || (filesStillToProcess > 0)){

        double matrixDeterminant;

        /** Retrieve Value Matrix */
        val = getMatrixVal (id);

        printf("Worker %u : Obtained Matrix %u.\n",id,val.id);

        /** Apply Gaussian Elimination */
        gaussianElimination(val.orderMatrix,val.matrix);

        /** Calculate Matrix Determinant */
        matrixDeterminant=calculateMatrixDeterminant(val.orderMatrix,val.matrix);

        /** Matrix Determinant Result */
        struct Matrix_result matrix_determinant_result;

        matrix_determinant_result.fileid=val.fileid;

        matrix_determinant_result.id=val.id;

        matrix_determinant_result.determinant=matrixDeterminant;

        printf("Worker %u : Determinant Calculated.\n",id);
        //usleep((unsigned int) floor (90000.0 * random () / RAND_MAX + 1.5));                           /* do something else */

        /** Store Matrix Determinant Result */
        putResults(matrix_determinant_result,id);

        printf("Worker %u : Saved Results obtained Matrix %u.\n",id,val.id);
        printf("MatrixProcessed - %u : Files To Process - %u\n",matrixProcessed,filesStillToProcess);
    };

    statusWorks[id] = EXIT_SUCCESS;
    pthread_exit (&statusWorks[id]);
}

static void printUsage (char *cmdName)
{
    fprintf (stderr, "\nSynopsis: %s OPTIONS [filename/number]\n"
                     " OPTIONS:\n"
                     " -t --- number of workers\n"
                     " -h --- print this help\n"
                     " -f --- filename\n"
                     , cmdName);
}
