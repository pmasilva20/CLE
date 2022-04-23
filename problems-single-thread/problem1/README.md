# Run problem1
gcc problem1_main.c problem1_functions.h problem1_functions.c prob1_processing.c prob1_processing.h -Wall -Wextra -Werror -O2 -std=c99 -pedantic -o countWords
# Run problem1 testcase
gcc test.c problem1_functions.h problem1_functions.c preprocessing.c preprocessing.h -Wall -Wextra -Werror -O2 -std=c99 -pedantic -o test