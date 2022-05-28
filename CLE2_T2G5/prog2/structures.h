/**
 *  \file structures.h
 *
 *  \brief Assignment 2 : Problem 2 - Determinant of a Square Matrix
 *
 *  Definition of Structures
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#ifndef PROG2_STRUCTURES_H
#define PROG2_STRUCTURES_H

/** Structure of Matrix processed Result */
struct MatrixResult{
    int id;
    int fileid;
    double determinant;
};

/** Structure of File */
struct FileMatrices{
    int id;
    FILE *pFile;
    char name[40];
    int numberOfMatrices;
    int orderOfMatrices;
    struct MatrixResult *determinant_result;
};

/** Structure of Matrix to process */
struct Matrix{
    int orderMatrix;
    int fileid;
    int id;
    double matrix[256][256];
};


#endif //PROG2_STRUCTURES_H