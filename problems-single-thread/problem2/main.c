#include <stdio.h>
#include "./problem2Operations.h"
#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <string.h>

static void printUsage(char *cmdName);

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

    if (argc == 1){
        fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
        printUsage (basename (argv[0]));
        return EXIT_FAILURE;
    }

    int error_code = processMatricesFile(fName);

    if (error_code != 0) {
        printf("Error during file processing of %s", fName);

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
