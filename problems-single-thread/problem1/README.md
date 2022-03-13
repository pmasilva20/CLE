# Run problem1
gcc problem1.c problem1_functions.h problem1_functions.c preprocessing.c preprocessing.h -Wall -Wextra -Werror -O2 -std=c99 -pedantic -o prob1
# Run problem1 testcase
gcc test.c problem1_functions.h problem1_functions.c preprocessing.c preprocessing.h -Wall -Wextra -Werror -O2 -std=c99 -pedantic -o test1