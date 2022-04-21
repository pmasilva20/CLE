#ifndef PROG2_STRUCTURES_H
#define PROG2_STRUCTURES_H

/** Structure of Matrix processed Result */
struct Matrix_result{
    int id;
    int fileid;
    double determinant;
};

/** Structure of File */
struct File_matrices{
    int id;
    char name[40];
    int numberOfMatrices;
    struct Matrix_result *determinant_result;
};

/** Structure of Matrix to process */
struct Matrix{
    int orderMatrix;
    int fileid;
    int id;
    double matrix[256][256];
};


#endif //PROG2_STRUCTURES_H