#ifndef GS_HELPER_H
#define GS_HELPER_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include "sys/time.h"

#define MAXIT  1000       // Maxinum number of iterations to allow
#define SEED   1486706860 // Random seed for matrix generation
#define MAGIC  2.324645f  // Arbitrary number designating unassigned value

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define IND(i,j,size) ((i)*(size)+(j))

void gauss_jacobi_pipelined(int num_threads, int asize, int tileSizeX, int tileSizeY);
void gauss_jacobi_sequential(int asize);

// Get the current time
double get_time();

// Record the execution time of one of the Gauss-Jacobi functions
void time_seqn(int asize);

// Record the execution time of one of the Gauss-Jacobi functions
void time_pipe(int threads, int asize, int tileSizeX, int tileSizeY);

// Allocate memory for and initiallize arrays
void init_arrays(int seed, int padded_asize, float **A);

// Deallocate memory for arrays
void clean_arrays(float **A);

#endif //#GS_HELPER_H