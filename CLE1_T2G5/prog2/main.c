/**
 *  \file main.c
 *
 *  \brief Assignment 1 : Problem 2 - Determinant of a Square Matrix
 *
 *  Main Program
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include "sharedRegion.h"
#include <string.h>
#include "structures.h"
#include "probConst.h"
#include <pthread.h>
#include <time.h>


/**
 * Calculate Matrix Determinant
 * \param size Order of the Matrix
 * \param matrix Matrix
 * \return Determinant of the Matrix after Gaussian Elimination
 */
double calculateMatrixDeterminant(int orderMatrix,double matrix[orderMatrix][orderMatrix]){

    /** \brief Apply Gaussian Elimination
     *  Generic square matrix of order n into an equivalent upper triangular matrix
    */
    for(int i=0;i<orderMatrix-1;i++){
        //Begin Gauss Elimination
        for(int k=i+1;k<orderMatrix;k++){
            double term=matrix[k][i]/matrix[i][i];
            for(int j=0;j<orderMatrix;j++){
                matrix[k][j]=matrix[k][j]-term*matrix[i][j];
            }
        }
    }

    double determinant=1;

    for (int x = 0; x < orderMatrix; x++){
        determinant*= matrix[x][x];
    }

    return determinant;

}

static void printUsage(char *cmdName);

/** \brief worker life cycle routine */
static void *worker (void *id);

/** \brief consumer threads return status array */
int *statusWorks;

/** \brief Number of Matrices processed by workers **/
int  matrixProcessed = 0;

/** \brief Number of Files to be processed **/
int filesToProcess =0;

int main(int argc, char** argv) {

    /** time limits **/
    struct timespec start, finish;

    /** \brief Number of Workers **/
    int numberWorkers=0;

    /** \brief  List of Files**/
    struct FileMatrices listFiles[N];

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
                if (optarg[0] == '-') {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                /** Initialization FileMatrices*/
                struct FileMatrices file_info;
                /** Set Name*/
                strcpy(file_info.name, optarg);
                /** Set id*/
                file_info.id=fileid;
                /** Save in List of Files*/
                listFiles[fileid]=file_info;
                /** Update fileid */
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

    if (argc == 1){
        fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
        printUsage (basename (argv[0]));
        return EXIT_FAILURE;
    }


    /** \brief Files to Process */
    filesToProcess=fileid;

    statusWorks = malloc(sizeof(int)*numberWorkers);

    /** Workers internal thread id array */
    pthread_t tIdWorkers[numberWorkers];

    /** Workers application defined thread id array*/
    unsigned int works[numberWorkers];

    /** Pointer to execution status */
    int *status_p;

    for (int i = 0; i < numberWorkers; i++)
        works[i] = i;

    srandom ((unsigned int) getpid ());


    /** Begin of Time measurement */
    clock_gettime (CLOCK_MONOTONIC_RAW, &start);

    /**
     * File Processing to save in Shared Region
     */
    for (int i = 0; i < filesToProcess; i++) {

        listFiles[i].pFile= fopen(listFiles[i].name, "r");

        if (!listFiles[i].pFile) {
            printf("\nError reading File: %s\n",listFiles[i].name);
        }
        else{

            if(fread(&listFiles[i].numberOfMatrices, sizeof(int), 1, listFiles[i].pFile)==0){
                printf("Main: Error reading Number of Matrices\n");
            }

            if(fread(&listFiles[i].orderOfMatrices, sizeof(int), 1, listFiles[i].pFile)==0){
                printf("Main: Error reading Order of Matrices\n");
            }

            printf("\nFile %u - Number of Matrices to be read  = %d\n", listFiles[i].id, listFiles[i].numberOfMatrices);

            printf("File %u - Order of Matrices to be read  = %d\n", listFiles[i].id, listFiles[i].orderOfMatrices);

            listFiles[i].determinant_result = malloc(sizeof(struct MatrixResult) * listFiles[i].numberOfMatrices);
        }
        /** Save File in Shared Region */
        putFileInfo(listFiles[i]);
    }


    /**
     * Generation of intervening Workers threads
     */
    for (int i = 0; i < numberWorkers; i++) {
        if (pthread_create(&tIdWorkers[i], NULL, worker, &works[i]) !=0)
        {
            perror("error on creating thread worker");
            exit(EXIT_FAILURE);
        }
        else{
            printf("Thread Worker Created %d !\n", i);
        }
    }

    /**
     * Read Successfully Files and send Matrices to store in the Shared Region
     */
    for (int id = 0; id < filesToProcess; id++) {

        if (listFiles[id].pFile) {

            for (int i = 0; i < listFiles[id].numberOfMatrices; i++) {

                struct Matrix matrix1;

                matrix1.fileid = listFiles[id].id;
                matrix1.id = i;
                matrix1.orderMatrix = listFiles[id].orderOfMatrices;

                if(fread(&matrix1.matrix, sizeof(double), listFiles[id].orderOfMatrices * listFiles[id].orderOfMatrices, listFiles[id].pFile)==0){
                    perror("Main: Error reading Matrix\n");
                }
                putMatrixVal(matrix1);

                printf("Main : Matrix %u to Shared Region.\n", i);
            };

            fclose(listFiles[id].pFile);
        }
    }


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

    /** End of measurement */
    clock_gettime (CLOCK_MONOTONIC_RAW, &finish);

    /** Print Final Results  */
    PrintResults(filesToProcess);

    /** Print Elapsed Time */
    printf ("\nElapsed time = %.6f s\n", (finish.tv_nsec - start.tv_nsec) / 1000000000.0);
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

    /** While there are matrix to process */
    while (getMatrixVal(id,&val)!=1){

        /** Matrix Determinant Result */
        struct MatrixResult matrix_determinant_result;

        /** Retrieve Value Matrix */
        printf("Worker %u : Obtained Matrix %u.\n",id,val.id);

        matrix_determinant_result.fileid=val.fileid;

        matrix_determinant_result.id=val.id;

        /** Calculate Matrix Determinant */
        matrix_determinant_result.determinant=calculateMatrixDeterminant(val.orderMatrix,val.matrix);

        printf("Worker %u : Matrix %u Determinant Obtained.\n",id,val.id);

        /** Store Matrix Determinant Result */
        putResults(matrix_determinant_result,id);

        printf("Worker %u : Saved Results obtained from Matrix %u.\n",id,val.id);
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
