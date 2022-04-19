#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include "structures.h"
#include "./problem1_functions.h"
#include "func.h"




/** \brief worker life cycle routine */
static void *worker (void *id);

/** \brief consumer threads return status array */
int *statusWorks;


static void printUsage (char *cmdName);


int main (int argc, char** argv){
    int opt;
    int index;
    int fCount = 0;
    int numberWorkers = 1;

    char* fName = "no name";
    char* files[argc];
    char* next;
  
    double time0, time1, timeTotal;

    timeTotal = 0.0;
  
   // int opterr = 0;
    do{
        switch ((opt = getopt(argc, argv, "f:h"))) {
            case 'f': /* file name */
                index = optind - 1;
                while(index < argc){
                    next = strdup(argv[index]);
                    index++;
                    if(next[0] != '-'){
                      files[fCount++] = next;  
                    }
                    else break;
                }
                break;


                if (optarg[0] == '-') {
                    fprintf(stderr, "%s: file name is missing\n", basename(argv[0]));
                    printUsage(basename(argv[0]));
                    return EXIT_FAILURE;
                }
                fName = optarg;
                printf("Read %s\n",fName);
                //int fnstart = optind - 1;


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

    } while(opt != -1);

    if (argc == 1){
        fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
        printUsage (basename (argv[0]));
        return EXIT_FAILURE;
    }

    printf("%s\n",files[0]);
    makeChunks("../text0.txt",10,100);


      //Make N worker threads
//    statusWorks = malloc(sizeof(int)*numberWorkers);
//
//    pthread_t tIdWorkers[numberWorkers];
//
//    unsigned int works[numberWorkers];
//
//    int *status_p;
//
//    for (int i = 0; i < numberWorkers; i++)
//        works[i] = i;
//
//    srandom ((unsigned int) getpid ());
//
//
//    //Inicializar Workers
//    for (int i = 0; i < numberWorkers; i++) {
//        if (pthread_create(&tIdWorkers[i], NULL, worker, &works[i]) !=0)
//        {
//            perror("error on creating thread worker");
//            exit(EXIT_FAILURE);
//        }
//        else{
//            printf("Thread Worker Created %d !\n", i);
//        }
//    }


    




















    //Iterate though files and do prob1
    for(int textIdx = 0; textIdx < fCount; textIdx++){
        //Vars needed
        int nWords = 0;
        int nVowelStartWords = 0;
        int nConsonantEndWord = 0;

        time0 = (double) clock() / CLOCKS_PER_SEC;

        //int error_code = problem1(argv[textIdx],&nWords,&nVowelStartWords,&nConsonantEndWord);
        
        
        
        time1 = (double) clock() / CLOCKS_PER_SEC;
        timeTotal += (time1 - time0);


        printf("File %s\n",files[textIdx]);
        printf("Number of words:%d\n",nWords);
        printf("Number of words which start with a vowel:%d\n",nVowelStartWords);
        printf("Number of words which end with a consonant:%d\n",nConsonantEndWord);
    }
}






static void *worker (void *par)
{
    unsigned int id = *((unsigned int *) par);                                                          /* consumer id */
    
    struct Chunk_text var;

    //while chunks to process
        //do prob1Funcs
        //save to SH
        //die
    
    // struct Matrix val;                                                                                /* produced value */

    // do{
    //     double matrixDeterminant;

    //     val = getMatrixVal (id);

    //     printf("Worker %u : Obtained Matrix.\n",id);

    //     gaussianElimination(val.orderMatrix,val.matrix);

    //     matrixDeterminant=calculateMatrixDeterminant(val.orderMatrix,val.matrix);

    //     struct Matrix_result matrix_determinant_result;

    //     matrix_determinant_result.fileid=val.fileid;

    //     matrix_determinant_result.id=val.id;

    //     matrix_determinant_result.determinant=matrixDeterminant;

    //     printf("Worker %u : Determinant Calculated.\n",id);

    //     putResults(matrix_determinant_result,id);

    //     printf("Worker %u : Saved Results obtained.\n",id);


    // } while ((matrixProcessed<matrixToProcess) || (filesStillToProcess > 0));

    // statusWorks[id] = EXIT_SUCCESS;
    // pthread_exit (&statusWorks[id]);
}



static void printUsage (char *cmdName)
{
    fprintf (stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                     " OPTIONS:\n"
                     " -h --- print this help\n"
                     " -f --- filename\n"
                     , cmdName);
}