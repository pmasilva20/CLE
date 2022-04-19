#include "./problem1_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <locale.h>
#include <stdbool.h>
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>

static void printUsage (char *cmdName);


int main (int argc, char** argv){
    int opt;
    int index;
    int fCount = 0;
    char* fName = "no name";
    char* files[argc];
    char* next;
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


    // int error_code = processMatricesFile(fName);

    // if (error_code != 0) {
    //     printf("Error during file processing of %s", fName);

    // }

    // return EXIT_SUCCESS;

    for(int textIdx = 0; textIdx < fCount; textIdx++){
        //Vars needed
        int nWords = 0;
        int nVowelStartWords = 0;
        int nConsonantEndWord = 0;

        int error_code = problem1(files[textIdx],&nWords,&nVowelStartWords,&nConsonantEndWord);

        if(error_code != 0){
            printf("Error during file processing of %s",files[textIdx]);
            continue;
        }

        printf("File %s\n",files[textIdx]);
        printf("Number of words:%d\n",nWords);
        printf("Number of words which start with a vowel:%d\n",nVowelStartWords);
        printf("Number of words which end with a consonant:%d\n",nConsonantEndWord);
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