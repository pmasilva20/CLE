# Problem 1
- Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
### How to Compile
``
gcc -Wall -O3 -o prog1 main.c assign1_functions.c prob1_processing.c shared_region.c assign1_worker.c -lpthread -lm
``

### Usage examples
``
./prog1 -f text0.txt -f text1.txt -f text2.txt -f text3.txt -f text4.txt -t 5
``
\
``
./prog1 -f text0.txt -t 10
``