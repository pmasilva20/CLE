/**
 *  \file sharedRegion.c
 *
 *  \brief Assignment 2 : Problem 2 - Determinant of a Square Matrix
 * *
 *  Shared Region
 *
 *  Dispatcher Operations:
 *     \li getMatrixVal
 *     \li putMatrixVal
 *
 *  \author João Soares (93078) e Pedro Silva (93011)
 */

#ifndef FIFO_H
#define FIFO_H

#include <stdbool.h>
#include "structures.h"

/**
 *  \brief Get a Matrix value from the data transfer region.
 *
 *  Operation carried out by the Thread Send Matrices and Receive Results.
 *
 *  \param consId thread identification
 *  \param *matrix Address of Variable Matrix
 *
 *  \return Whatever there is Work to do;
 */
int getMatrixVal(unsigned int consId,struct Matrix *matrix);

/**
 *  \brief Store a Matrix value in the data transfer region.
 *
 *  Operation carried out by the Thread Read Matrices of the File.
 *
 *  \param consId thread identification
 *  \param val Matrix to be stored
 */
extern void putMatrixVal(unsigned int consId,struct Matrix matrix);

#endif
