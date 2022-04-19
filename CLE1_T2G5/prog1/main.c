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
#include "fifo.h"
#include "preprocessing.h"
#include <stdbool.h>
#include <ctype.h>


/** \brief worker life cycle routine */
static void *worker (void *id);

/** \brief consumer threads return status array */
int *statusWorks;

/** \brief Number of Chunks to be processed **/
int chunksToProcess =0;


static void printUsage (char *cmdName);


int main (int argc, char** argv){
    int opt;
    int index;
    int fCount = 0;
    int numberWorkers = 2;

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
    int chunksCount = makeChunks("../text0.txt",10,20);
    chunksToProcess += chunksCount;


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

    //Join all threads and then get all results from SR
    for(int i = 0; i < numberWorkers; i++){
        if (pthread_join (tIdWorkers[i], (void *) &status_p) != 0){
            perror ("error on waiting for thread producer");
            exit (EXIT_FAILURE);
        }
        printf ("thread worker, with id %u, has terminated: ", i);
        printf ("its status was %d\n", *status_p);
    }


}






static void *worker (void *par)
{
    unsigned int id = *((unsigned int *) par);                                                          /* consumer id */

    printf("Soldier %d!\n",id);
    struct Chunk_text var;

    //while chunks to process
        //do prob1Funcs
        //save to SH
        //die

    do {
        //Get chunk
        struct Chunk_text chunk = getChunkText();
        //Do prob1 processing
        int nWords = 0;
        int nVowelStartWords = 0;
        int nConsonantEndWord = 0;

        //State Flags
        bool inWord = false;

        //Read files
        int character;
        int previousCharacter = 0;

        for(int i = 0; i < chunk.count; i++){
            character = chunk.chunk[i];
            //Check if inWord
            if(inWord){
                //If white space or separation or punctuation simbol -> inWord is False
                //if lastchar is consonant
                if(checkForSpecialSymbols(character)){
                    inWord = false;
                    if(checkConsonants(previousCharacter)){
                        nConsonantEndWord+=1;
                    }
                }
                //If alphanumeric character or underscore or apostrophe -> nothing
                //lastChar = character
                if(isalnum(character) ||  character == '_' || character == '\''
                   || character == 0xE28098 || character == 0xE28099){
                    previousCharacter = character;
                }
            }
            else{
                //If white space or separation or punctuation simbol -> nothing
                //If alphanumeric character or underscore or apostrophe -> inWord is True
                //nWords += 1, checkVowel() -> nWordsBV+=1, lastChar = character
                if(isalnum(character) ||  character == '_' || character == '\''
                   || character == 0xE28098 || character == 0xE28099){
                    inWord = true;
                    nWords +=1;
                    if(checkVowels(character)){
                        nVowelStartWords+=1;
                    }
                    previousCharacter = character;
                }
            }
        }
        printf("Chunk %d\n",id);
        printf("Nwords %d\n",nWords);
        printf("NVowelwords %d\n",nVowelStartWords);
        printf("NConsonantswords %d\n",nConsonantEndWord);

        printf("Remaining chunksToProcess %d\n",chunksToProcess);
        printf("\n");
        chunksToProcess--;
    } while (chunksToProcess > 0);

    statusWorks[id] = EXIT_SUCCESS;
    printUsage("Exiting\n");
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