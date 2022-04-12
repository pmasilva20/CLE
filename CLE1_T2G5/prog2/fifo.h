/**
 *  \file fifo.h (interface file)
 *
 *  \brief Problem name: Producers / Consumers.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  Data transfer region implemented as a monitor.
 *
 *  Definition of the operations carried out by the producers / consumers:
 *     \li putVal
 *     \li getVal.
 *
 *  \author António Rui Borges - March 2019
 */

#ifndef FIFO_H
#define FIFO_H

#include "structures.h"

/**
 *  \brief Store a value in the data transfer region.
 *
 *  Operation carried out by the producers.
 *
 *  \param prodId producer identification
 *  \param val value to be stored
 */

//extern void putVal (unsigned int prodId, unsigned int val);

/**
 *  \brief Get a value from the data transfer region.
 *
 *  Operation carried out by the consumers.
 *
 *  \param consId consumer identification
 *
 *  \return value
 */

//extern unsigned int getVal (unsigned int consId);

struct Matrix getMatrixVal(unsigned int consId);

extern putMatrixVal(struct Matrix matrix);
extern void putResults(struct Matrix_result result,unsigned int consId);
extern void putFileInfo(struct File_matrices file_info);
extern void getResults();
#endif /* FIFO_H */
