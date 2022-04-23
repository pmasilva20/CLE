/**
 *  \file prob1_processing.c
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
int detectBytesNeeded(int character){
    if(character < 192){
        return 0;
    }
    else if (character < 224)
    {
        return 2;
    }
    else if (character < 240)
    {
        return 3;
    }
    else
    {
        return 4;
    }
}
/**
 * Check if character is a continuation symbol that connects characters into a single word
 * @param character Character to check
 * @return True if a continuation symbol, false otherwise
 */
bool checkForContinuationSymbols(int character){
    if(character ==  0x27 
        || character == 0x60
        || character == 0xE28098 
        || character == 0xE28099)
        {
            return true;
        }
        return false;
}

/**
 * Check if character is a special symbol that marks the end of a word
 * @param character Character to compare to
 * @return True if a special symbol, false otherwise
 */
bool checkForSpecialSymbols(int character){
   if(character == 0x20                                   // space
        || character == 0x9                               // \t
        || character == 0xA                               // \n
        || character == 0xD                               // \r
        || character == 0x5b                              // [
        || character == 0x5d                              // ]
        || character == 0x3f                              // ?
        || character == 0xc2ab                            // «
        || character == 0xc2bb                            // »
        || character == 0xe280a6                          // …
        || 0x21 == character                              // !
        || character == 0x22                              // "
        || 0x28 == character || character == 0x29         // ( )
        || (0x2c <= character && character <= 0x2e)       // , - .
        || 0x3a == character || character == 0x3b         // : ;
        || 0xe28093 == character || character == 0xe28094 // – —
        || 0xe2809c == character || character == 0xe2809d // “ ”
        ){
           return true;
        }
        return false;
}

/**
 * Checks if character is a vowel(including accented vowels)
 * @param character Character to compare to
 * @return True if a vowel, false otherwise
 */
int checkVowels(int character){
   if(character == 0x41                            // A
        || character == 0x45                         // E
        || character == 0x49                         // I
        || character == 0x4f                         // O
        || character == 0x55                         // U
        || character == 0x61                         // a
        || character == 0x65                         // e
        || character == 0x69                         // i
        || character == 0x6f                         // o
        || character == 0x75                         // u
        || (0xc380 <= character && character <= 0xc383) // À Á Â Ã
        || (0xc388 <= character && character <= 0xc38a) // È É Ê
        || (0xc38c == character || character == 0xc38d) // Ì Í
        || (0xc392 <= character && character <= 0xc395) // Ò Ó Ô Õ
        || (0xc399 == character || character == 0xc39a) // Ù Ú
        || (0xc3a0 <= character && character <= 0xc3a3) // à á â ã
        || (0xc3a8 <= character && character <= 0xc3aa) // è é ê
        || (0xc3ac == character || character == 0xc3ad) // ì í
        || (0xc3b2 <= character && character <= 0xc3b5) // ò ó ô õ
        || (0xc3b9 == character || character == 0xc3ba) // ù ú
        ){
                return true;
        }
        return false;
}
/**
 * Checks if character is a consonant(including the cedilla character)
 * @param character Character to compare to
 * @return True if a consonant, false otherwise
 */
int checkConsonants(int character){
    if(character == 0xc387                      // Ç
        || character == 0xc3a7                   // ç
        || (0x42 <= character && character <= 0x44) // B C D
        || (0x46 <= character && character <= 0x48) // F G H
        || (0x4a <= character && character <= 0x4e) // J K L M N
        || (0x50 <= character && character <= 0x54) // P Q R S T
        || (0x56 <= character && character <= 0x5a) // V W X Y Z
        || (0x62 <= character && character <= 0x64) // b c d
        || (0x66 <= character && character <= 0x68) // f g h
        || (0x6a <= character && character <= 0x6e) // j k l m n
        || (0x70 <= character && character <= 0x74) // p q r s t
        || (0x76 <= character && character <= 0x7a) // v w x y z
        ){
                return true;
        }
        return false;
}
