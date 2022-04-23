/**
 *  \file prob1_processing.h
 *
 *  \brief Assignment 1 : Problem 1 - Number of Words, Number of Words starting with a Vowel and Number of Words ending with a Consonant
 *
 *  Functions used for Problem 1
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#include <stdbool.h>

/**
 * \brief Detect how many bytes need to be read for UTF-8 by taking into account first byte
 * @param character First byte of UTF-8 character
 * @return Number of bytes that need to be read not including the first byte
 */
int detectBytesNeeded(int character);
/**
 * Checks if character is a consonant(including the cedilla character)
 * @param character Character to compare to
 * @return True if a consonant, false otherwise
 */
int checkConsonants(int character);
/**
 * Checks if character is a vowel(including accented vowels)
 * @param character Character to compare to
 * @return True if a vowel, false otherwise
 */
int checkVowels(int character);
/**
 * Check if character is a continuation symbol that connects characters into a single word
 * @param character Character to check
 * @return True if a continuation symbol, false otherwise
 */
bool checkForContinuationSymbols(int character);
/**
 * Check if character is a special symbol that marks the end of a word
 * @param character Character to compare to
 * @return True if a special symbol, false otherwise
 */
bool checkForSpecialSymbols(int character);