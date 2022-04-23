#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include "structures.h"
#include "func.h"
#include "fifo.h"
#include "prob1_processing.h"
#include "worker.h"


/** \brief consumer threads return status array */
int *statusWorks;



static void printUsage (char *cmdName);


int main (int argc, char** argv){
    int opt;
    int index;
    int fCount = 0;
    int numberWorkers = 10;

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


    //Make N worker threads
    statusWorks = malloc(sizeof(int)*numberWorkers);

    pthread_t tIdWorkers[numberWorkers];

    //Id's
    unsigned int works[numberWorkers];

    int *status_p;

    for (int i = 0; i < numberWorkers; i++)
        works[i] = i;

    srandom ((unsigned int) getpid ());


    //Inicializar Workers
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





    //TODO:Look at parameters
    makeChunks("../text4.txt",10,10);
    finishedProcessingChunks();


    //Join all threads and then get all results from SR
    for(int i = 0; i < numberWorkers; i++){
        if (pthread_join (tIdWorkers[i], (void *) &status_p) != 0){
            perror ("error on waiting for thread producer");
            exit (EXIT_FAILURE);
        }
        printf ("thread worker, with id %u, has terminated\n", i);
        //printf ("its status was %d\n", *status_p);
    }
    for(int i = 0; i < 1; i++){
        struct File_text* text = getFileText(10);
        if(text != NULL){
            printf("File:%d\n",(*text).fileId);
            printf("Words:%d\n",(*text).nWords);
            printf("Vowel words:%d\n",(*text).nVowelStartWords);
            printf("Consonant words:%d\n",(*text).nConsonantEndWord);
        }
        else printf("Error retrieving files statistics for file %d",10);

    }

}



static void printUsage (char *cmdName)
{
    fprintf (stderr, "\nSynopsis: %s OPTIONS [filename]\n"
                     " OPTIONS:\n"
                     " -h --- print this help\n"
                     " -f --- filename\n"
                     , cmdName);
}