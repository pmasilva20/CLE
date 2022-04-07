//
// Created by joaots on 07/04/22.
//
//TODO: Ver forma de alocar melhor a memória.

#ifndef PROG2_STRUCTURES_H
#define PROG2_STRUCTURES_H
struct Matrix_result{
    int id;
    double determinant;
};

struct File{
    int id;
    char name[40];
    struct Matrix_result determinant_result[256];
};

struct Matrix{
    int orderMatrix;
    int id;
    double matrix[256][256];
};

#endif //PROG2_STRUCTURES_H