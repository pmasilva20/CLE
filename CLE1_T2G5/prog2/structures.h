//
// Created by joaots on 07/04/22.
//
//TODO: Ver forma de alocar melhor a memória.

#ifndef PROG2_STRUCTURES_H
#define PROG2_STRUCTURES_H

//TODO: Tentar uns mallocs

struct Matrix_result{
    int id;
    int fileid;
    double determinant;
};

struct File_matrices{
    int id;
    char name[40];
    int numberOfMatrices;
    struct Matrix_result *determinant_result;
};

struct Matrix{
    int orderMatrix;
    int fileid;
    int id;
    double matrix[256][256]; //TODO: Ver como fazer  aqui o malloc;
};


#endif //PROG2_STRUCTURES_H