//
// Created by joaots on 07/04/22.
//
#include <stdio.h>
#include "structures.h"
#include <stdlib.h>



void print_matrix_details(struct Matrix matrix){
    printf("Fileid: %d\n",matrix.fileid);
    printf("Matrixid: %d\n",matrix.id);
    printf("Order Matrix: %d\n",matrix.orderMatrix);
}


