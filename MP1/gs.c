#include "gs-helper.h"
#include <math.h>
#include "sys/time.h"

double getclock() {
    struct timeval tv;
    int ok;
    ok=gettimeofday(&tv,0);
    if(ok<0) {
        printf("error");
        exit(-1);
    }
    return (tv.tv_sec*1.0+tv.tv_usec*1.0E-6);
}

// Comment/uncomment the below line to hide/show arrays
#define DPRINT
// Sequential Gauss-Jacobi solver using the red-black method
void gauss_jacobi_sequential(int asize) {
  int p_asize = asize+2; // Padded array size (+1 padding to each side)
  float *A, maxError, oldValue;
  init_arrays(SEED,p_asize,&A);
  
  int it, i, j;
  for(it = 1; it <= 1; ++it) { // Repeat the estimation process
    // Red
    bool alt = false;
    for (i = 1; i < p_asize-1; i++) {
      for (j = !alt ? 1 : 2; j < p_asize-1; j+=2) {
        A[IND(i,j,p_asize)] = 0.25*(
        A[IND(i-1,j,p_asize)] +
        A[IND(i+1,j,p_asize)] +
        A[IND(i,j-1,p_asize)] +
        A[IND(i,j+1,p_asize)]);
      }
      alt = !alt;
    }
    
    // Black
    alt = true;
    for (i = 1; i < p_asize-1; i++) {
      for (j = !alt ? 1 : 2; j < p_asize-1; j+=2) {
        A[IND(i,j,p_asize)] = 0.25*(
        A[IND(i-1,j,p_asize)] +
        A[IND(i+1,j,p_asize)] +
        A[IND(i,j-1,p_asize)] +
        A[IND(i,j+1,p_asize)]);
      }
      alt = !alt;
    }
  }
  #ifdef DPRINT
    for(i = 0; i < p_asize; i+=p_asize/5) {
      printf("A[%4d,%4d]=%7.10f\n", i, i, A[IND(i,i,p_asize)]);
    }
  #endif

  clean_arrays(&A);
}

// Parallel Gauss-Jacobi solver using a spinlock (signal wait)
void gauss_jacobi_pipelined(int num_threads, int asize, int tile_size_x, int tile_size_y) {
  float *A;
  int p_asize = asize+2; // Padded array size (+1 padding to each side)
  init_arrays(SEED, p_asize, &A);
  int i,j,k;
  /* ****************** your code goes here ****************** */
  //calculate start and;
  int block_num;
  block_num=(asize*asize)/(tile_size_x*tile_size_y);
   
 // int x_start[block_num],x_end[block_num],y_start[block_num],y_end[block_num];
  int* x_start = malloc(block_num*sizeof(int));
  int* x_end = malloc(block_num*sizeof(int));
  int* y_start = malloc(block_num*sizeof(int));
  int* y_end = malloc(block_num*sizeof(int));
  x_start[0]=1;
  y_start[0]=1;
  for(i=0;i<block_num;i++) {
    
    if(i==0) {
      x_end[i]=x_start[0]+tile_size_x-1;
      y_end[i]=y_start[0]+tile_size_y-1; 

    }   
    else {
      if(x_end[i-1]==asize) {  
        x_start[i]=1;
        x_end[i]=x_start[i]+tile_size_x-1;
        y_start[i]=y_end[i-1]+1;
        y_end[i]=y_start[i]+tile_size_y-1;
      }
      else { 
        x_start[i]=x_end[i-1]+1;
        x_end[i]=x_start[i]+tile_size_x-1;
        y_start[i]=y_start[i-1];
        y_end[i]=y_end[i-1];
      }
    }
  }
  
  int **syn_table=malloc(p_asize*sizeof(int*));
  for(i=0;i<p_asize;i++) {
    syn_table[i]=malloc(p_asize*sizeof(int));
  }
  //initialize syn_table
  for(i=0;i<p_asize;i++) {
    for(j=0;j<p_asize;j++) {
        syn_table[i][j]=0;
    }
  }
  for(i=0;i<p_asize;i++) {
    syn_table[i][0]=MAXIT;
    syn_table[i][p_asize-1]=MAXIT;
  }
  for(i=0;i<p_asize;i++) {
    syn_table[0][i]=MAXIT;
    syn_table[p_asize-1][i]=MAXIT;
  } 
    int current_thread_id;
    int current_x_start;
    int current_x_end;
    int current_y_start;
    int current_y_end;
    int it,r,t;
 
  #pragma omp parallel  num_threads(num_threads) shared(syn_table) private(current_thread_id,current_x_start,current_x_end,current_y_start,current_y_end)
  {
    int it,r,t;
    double t1,t2;
    current_thread_id=omp_get_thread_num();
    current_x_start=x_start[current_thread_id];
    current_x_end=x_end[current_thread_id];
    current_y_start=y_start[current_thread_id];
    current_y_end=y_end[current_thread_id];
   // printf("thread id:%d, x_strat:%d x_end:%d y_start:%d y_end:%d\n",current_thread_id,current_x_start,current_x_end,current_y_start,current_y_end);
    //red
for(it=1;it<=1;it++) {

    bool alt=false;
    for(r=current_x_start;r<=current_x_end;r++) {
      for (t = !alt ? current_y_start : current_y_start +1 ; t <= current_y_end ; t+=2) {
          while( syn_table[r][t] > syn_table[r-1][t] || syn_table[r][t] > syn_table[r][t-1] || syn_table[r][t] > syn_table[r+1][t] || syn_table[r][t] >syn_table[r][t+1]  ) {
            #pragma omp flush(syn_table)  
            }
          A[IND(r,t,p_asize)] = 0.25*(
          A[IND(r-1,t,p_asize)] +
          A[IND(r+1,t,p_asize)] +
          A[IND(r,t-1,p_asize)] +
          A[IND(r,t+1,p_asize)]);
          syn_table[r][t]+=1;
	  #pragma omp flush(A)
          #pragma omp flush(syn_table)  
        }
        alt = !alt;     
    }
    //black
    alt=true;
    for(r=current_x_start;r<=current_x_end;r++) {
        for(t=!alt?current_y_start : current_y_start+1;t<=current_y_end;t+=2) {
            while( syn_table[r][t] >= syn_table[r][t-1] || syn_table[r][t] >= syn_table[r][t+1] || syn_table[r][t] >= syn_table[r-1][t] || syn_table[r][t] >= syn_table[r+1][t]  ) { 
            #pragma omp flush(syn_table)
             }
            A[IND(r,t,p_asize)] = 0.25*(
            A[IND(r-1,t,p_asize)] +
            A[IND(r+1,t,p_asize)] +
            A[IND(r,t-1,p_asize)] +
            A[IND(r,t+1,p_asize)]);
            syn_table[r][t]+=1;
            #pragma omp flush(syn_table)
	    #pragma omp flush(A)
        }
        alt=!alt;
    }
  }//iteration
} //pragma omp parallel
  #ifdef DPRINT
    for(i = 0; i < p_asize; i+=p_asize/5) {
      printf("A[%4d,%4d]=%7.10f\n", i, i, A[IND(i,i,p_asize)]);
    }
  #endif

  clean_arrays(&A);
}

int main(int argc, char** argv) {
  int ncores = omp_get_max_threads();
  int num_threads = 1;
  int asize = 8;
  int tile_size_x = 4;
  int tile_size_y = 4;

  if (argc == 4) {
    asize = atoi(argv[1]);
    tile_size_x = atoi(argv[2]);
    tile_size_y = atoi(argv[3]);
  }

  num_threads = (asize*asize)/(tile_size_x*tile_size_y);

  // Run and time the sequential version
  time_seqn(asize);

  // Run and time the parallel version
  time_pipe(num_threads, asize, tile_size_x, tile_size_y);

  return 0;
}
