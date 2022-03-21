#include <stdbool.h>

int preprocessChar(int character){
    //Á,À,Ã,Â
    if(character >= 0xc380 && character <= 0xc383){
            character = 'a';
    }
    //á,ã,...
    else if (character >= 0xc3a0 && character <= 0xc3a3){
            character = 'a';
    }
    //é,è,...
    else if (character >= 0xc3a8 && character <= 0xc3aa){
            character = 'e';
    }
    //É,È,...
    else if (character >= 0xc388 && character <= 0xc38a){
            character = 'e';
    }
    //Í,Ì,...
    else if (character >= 0xc38c && character <= 0xc38e){
            character = 'i';
    }
    //í,ì,...
    else if (character >= 0xc3ac && character <= 0xc3ae){
            character = 'i';
    }
    //Ó,Ò,...
    else if (character >= 0xc392 && character <= 0xc395){
            character = 'o';
    }
    //ó,ò,...
    else if (character >= 0xc3b2 && character <= 0xc3b5){
            character = 'o';
    }
    //ú,ù,...
    else if (character >= 0xc3b9 && character <= 0xc3bb){
            character = 'u';
    }
    //Ú,Ù,...
    else if (character >= 0xc399 && character <= 0xc39b){
            character = 'u';
    }
    //ç,Ç,...
    else if (character == 0xc387 || character == 0xc3a7 ){
            character = 'c';
    }
    //All types of dash -> hyphen
    else if(character >= 0xe28090  && character <= 0xe28095){
        character = '-';
    }
    //Left and Right double quotation marks
    else if(character == 0xe2809c  || character == 0xe2809d){
        character = '"';
    }
    //Elipsis to Full point
    else if(character == 0xe280a6){
        character = '.';
    }
    //Left and Right single quotation marks
    else if(character == 0xe28098  || character == 0xe28099 || character == '`'){
        character = '\'';
    }
    return character;
}

int checkVowels(int character){
    switch (character)
    {
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
        return true;
    }
    return false;
}

int checkConsonants(int character){
    if(!checkVowels(character) && character >= 'a' && character <= 'z'){
        return true;
    }
    return false;
}


int checkUpperCase(int character){
    //If Uppercase
    if (character >= 65 && character <= 90){
        character = character +32;
        return character;
    }
    else{
        return  character;
    }
}