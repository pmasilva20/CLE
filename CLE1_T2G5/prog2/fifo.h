#ifndef FIFO_H
#define FIFO_H

#include <stdbool.h>
#include "structures.h"

/**
 *  \brief Get a Matrix value from the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param consId worker identification
 *  \param *matrix Address of Variable Matrix
 *
 *  \return Whatever there is Work to do;
 */
int getMatrixVal(unsigned int consId, struct Matrix *matrix);

/**
 *  \brief Store a Matrix value in the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param prodId worker identification
 *  \param val Matrix to be stored
 */
extern void putMatrixVal(struct Matrix matrix);

/**
 *  \brief Store a Determinant value of Matrix in the data transfer region.
 *
 *  Operation carried out by the workers.
 *
 *  \param prodId worker identification
 *  \param val Determinant value of Matrix to be stored
 */
extern void putResults(struct Matrix_result result,unsigned int consId);


/**
 *  \brief Store a File (File_matrices) value in the data transfer region.
 *
 *  Operation carried out by the Main.
 *
 *  \param val File (File_matrices) to be stored
 */
extern void putFileInfo(struct File_matrices file_info);

/**
 *  \brief Print in the terminal the results stored in the Shared Region
 *  \param filesToProcess Number of Files
 */
extern void getResults(int filesToProcess);

#endif
