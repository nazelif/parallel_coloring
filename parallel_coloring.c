#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

typedef int bool;
#define true 1
#define false 0

/* Modified the read_graph and write_file functions from github.com/sdasgup3/parallel-sudoku/tree/master/MpiVersion */

#define DEBUG_VERTEX_DISTRIBUTION 0
#define DEBUG_PARALLEL_COLORING 0

char * INPUT_PATH = "ENS_CS249_FinalProject/";
char * OUTPUT_PATH = "ENS_CS249_FinalProject/";
int rank, num_processors, root = 0; 
int V, E; //number of vertices and edges
int chromaticity_upper = -1; //upper bound on the chromaticity of the graph, initialized to -1
//chromaticity = minimum number of colors required to color the graph
int *  adjmatrix; //the symmetric adjacency graph matrix.  
// The index of element at col,row is idx = row * V + col. 
// The element at index i is at row = i/V, col = i % V
int * colors; //the colors assigned to each vertex
int * boundary_matrix;

// Functions defined below
int compare (const void *a, const void *b);
void read_graph (char * filename);
void write_colors (char * filename);
void parallel_coloring(int* range,int * offsets,int * p_graph);
void printMatrix(int row, int col, int* matrix);
void printArray(int row, int* array);
void create_boundary_matrix (int* matrix);

int main (int argc, char** argv)
{
  char *input_filename, *output_filename;
  int num_v_per_p, remainder_v_per_p; 
  int first_v, last_v, *range; //id's of the first and last vertices in a given processor
  int maxColor; 
  int i, j, k;
  int *p_graph, *p_graph_size, *offsets;//, *boundary_graph;
  //graph with edges corresponding to the vertices in this process and their size. 
  //If process p has vertices 0 and 1, then p_graph will be a 2xV matrix, so it has all the edges between 1,2 
  // and all vertices offsets is the index in graph where to start copying to p_graph
  
  int *vertex_offsets;
  double start_time, end_time, runtime, largest_runtime;

  //Initialize MPI
  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processors);

  // Error check on the number of arguments, returns with an error
  if (argc != 3) { 
    if (rank == root)
      printf("Usage: mpirun/mpiexec -np xx ./graphcoloring input_filename output_filename\n");
    MPI_Finalize();
    return -1;
  }

  input_filename = malloc(900);
  strcpy(input_filename,INPUT_PATH);
  strcat(input_filename,argv[1]);

  output_filename = malloc(900);
  strcpy(output_filename,OUTPUT_PATH);
  strcat(output_filename,argv[2]);


#if DEBUG_VERTEX_DISTRIBUTION
  //every process reads the file and loads graph 
  read_graph(input_filename);
  printMatrix(V,V,adjmatrix);
  if (rank == root) {
    printf("V = %d E = %d max_degree = %d\n", V, E, chromaticity_upper);
    fflush(stdout);
  }
  
#else
  //only root reads the file and loads the full graph
  if (rank == root) {
    read_graph(input_filename);
    printf("V = %d E = %d Chromaticity Upper Bound = %d\n", V, E, chromaticity_upper);
    printf("Root finished reading the graph from file.\n");
    fflush(stdout);
  }

  //root broadcasts E,V and chromaticity_upper to all other processes
  MPI_Bcast(&V,1,MPI_INT,root,MPI_COMM_WORLD);
  MPI_Bcast(&E,1,MPI_INT,root,MPI_COMM_WORLD);
  MPI_Bcast(&chromaticity_upper,1,MPI_INT,root,MPI_COMM_WORLD);
#endif

// Distribution: create an initial adjacency matrix
  num_v_per_p = V/num_processors;
  p_graph_size = (int *) malloc(num_processors * sizeof(int));
  offsets = (int *)malloc(num_processors * sizeof(int));
  vertex_offsets = (int *)malloc(num_processors * sizeof(int));
  range = (int *)malloc(num_processors * sizeof(int));

  
  // More distribution
  for (i = 0; i < num_processors; i++) {  
    remainder_v_per_p = (V + i) % num_processors;
    first_v =  num_v_per_p * i + remainder_v_per_p * (remainder_v_per_p < i);     //index of the first vertex for process i
    last_v = (i + 1) * num_v_per_p + (remainder_v_per_p+1) * (remainder_v_per_p < i) - 1; //index of the last vertex for process i
    range[i] = last_v - first_v + 1;
    p_graph_size[i] = range[i] * V;
    
    offsets[0] = 0;
    vertex_offsets[0] = 0;

    if (i > 0) {
      offsets[i] = offsets[i-1] + p_graph_size[i-1];
      vertex_offsets[i] = vertex_offsets[i-1] + range[i-1];
    }
	
	#if DEBUG_VERTEX_DISTRIBUTION    
    printf("V = %d rank = %d first = %d last = %d range = %d ideal_range = %d\n",V,i,first_v,last_v,range[i],(V+i)/num_processors);
	#endif
  } // done with the processors

  p_graph = (int *) malloc(p_graph_size[rank] * sizeof(int));
  //boundary_graph = (int *) malloc(p_graph_size[rank] * sizeof(int));

  MPI_Scatterv(adjmatrix, p_graph_size, offsets, MPI_INT, p_graph, p_graph_size[rank], MPI_INT, root, MPI_COMM_WORLD);

  #if DEBUG_VERTEX_DISTRIBUTION
  //check whether p_graph and graph match in the corresponding positions
  for (i = 0; i < range[rank]; i++) {
    for (j = 0; j < V; j++) {
      if (p_graph[i*V + j] != graph[offsets[rank] + i * V + j])
	printf("Incorrect subgraph assignment in p_graph process %d at row %d col %d\n",rank, i, j);
    }
  }
  #endif

  if (rank == root) {
    boundary_matrix = (int *) malloc(V * V * sizeof(int));
    memset(boundary_matrix, -1, V * V * sizeof(int));
    create_boundary_matrix(vertex_offsets);
    printf("---Boundary Matrix---\n");
    //printf("x == -1 where no edges, x >= 0 where the processor to communicate to is known\n");
    printMatrix(V,V,boundary_matrix);

  }

  start_time = MPI_Wtime();

  //initialize colors to 0
  colors = (int *) malloc(V * sizeof(int));
  memset(colors, 0, V * sizeof(int));

  
  printf("p_graph of process #%d\n", rank);
  printArray(p_graph_size[rank], p_graph);
  
  /*
  printf("Vertex_offsets\n");
  printArray(num_processors,vertex_offsets);
  printf("Range\n");
  printArray(num_processors, range);
  */

  // INITIAL COLORING 
  parallel_coloring(range, vertex_offsets, p_graph);

  //printf("Printing p_graph\n");
  //printMatrix(V, num_v_per_p, p_graph);

  end_time = MPI_Wtime();
  runtime = end_time-start_time;

  //find the largest runtime (most likely it will be root's runtime since root
  //does a few extra things like generating weights, gathering colors and
  //synchronizing them)
  MPI_Allreduce(&runtime,&largest_runtime,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);

  printf("Max runtime was %f\n",largest_runtime);
 
  int num_uncolored =0;
  for (i=0;i<V;i++) { //find out how many vertices are uncolered
    if (colors[i] == 0)
      num_uncolored++;
  }
  if (num_uncolored > 0)
    printf("Not all vertices have been colored\n");

  if (rank == root){//root writes colors to file
    write_colors(output_filename);    
  }

  free(input_filename);
  free(output_filename);
  free(p_graph);
  free(range);
  free(p_graph_size);
  free(offsets);
  free(adjmatrix);
  free(colors);
  free(boundary_matrix);
  MPI_Finalize();  //Finalize
  return 0;
}

void create_boundary_matrix(int* vertex_offsets) {
  int x = num_processors-1; 
  
  int i, j, remainder;
  for (i = 0; i < V*V ; i++){
    if (adjmatrix[i] == 1){
      remainder = i % V;
      for (j = 0; j < num_processors; j++){
        if (remainder >= vertex_offsets[j] && (remainder < vertex_offsets[j+1] || j ==  x )){
          boundary_matrix[i] = j;  
        }
      }
    }
  }  
}


void parallel_coloring(int* range,int * offsets,int * p_graph)
{
  int i,j,k;
  int num_v_per_p, remainder_v_per_p, first_v;
  int * vtx_colors, * neighbor_colors;
  int num_colors, min_color;
  int procesor_to_communicate;

  num_v_per_p = V/num_processors;
  vtx_colors = (int *) malloc(range[rank] * sizeof(int));   
  memset(vtx_colors,0,range[rank] * sizeof(int));


  // while the chromacity is less than the upper bound
  for (i = 0; i < chromaticity_upper; i++) { 
    remainder_v_per_p = (V + rank) % num_processors; 
    first_v =  num_v_per_p * rank +  remainder_v_per_p * (remainder_v_per_p < rank); //index of the first vertex for process i
    for (j=0; j < range[rank]; j++) {   //for each vertex of this process
        neighbor_colors = (int *) malloc(V * sizeof(int));
        memset(neighbor_colors, 0, V * sizeof(int));
        num_colors = 0;

        for (k = 0; k < V; k++) {              
          if (p_graph[j * V + k] == 1) {//if there is an edge between j vertex and neighbor k vertex
            if (colors[k] != 0) { //if neighbor is colored just add its color to the neighbor_colors
                neighbor_colors[num_colors] = colors[k];
                num_colors++;
            } // if neighbor is not colored, its colored when its turn comes, code below
            else {
              colors[k] = min_color + 1;
              min_color++;
            }
          } // end if
        }// end of for k loop  
        
        if (colors[first_v + j] == 0){ // the vertex hasn't been colored
          // color with the smallest color that is not in the neighbor colors

          // sort the neighbor_colors array
          qsort(neighbor_colors, num_colors, sizeof(int), compare);    

          if (num_colors == 0 || neighbor_colors[0] > 1){ //if none of the neighbors is colored or the smallest color of a neighbor is >1
            min_color = 1;
          }
            
          else {
            for (k = 0;k < V; k++) {
              // THIS COLORS AT LEAST 278
              //min_color = neighbor_colors[k] + 1;
              
              //In between a color in the array of neighbors colors if there is a gap between two of the (sorted) neighbors colors
              if (k<V-1 && (neighbor_colors[k+1]-neighbor_colors[k]>1)) {
                 min_color = neighbor_colors[j-1] + 1;

                break;
              }
              else {
                min_color = neighbor_colors[num_colors-1] + 1;
              }
           }
        }
        //if vtx_colors[j-1]
        vtx_colors[j] = min_color; //color
          // printf("min_color %d\n", min_color);
          // printf("Process %d colored vertex %d with color %d\n", rank, j + first_v + 1, min_color);

          #if DEBUG_PARALLEL_COLORING
              if (i==1) {
                  printf("START DEBUG_PARALLEL_COLORING, its processor %d\n", rank);
                int m;
                  if (num_colors == 0)
                    printf("rank=%d j=%d color=%d\n",rank,j,min_color);
                    for (m = 0; m < num_colors; m++) {
                      printf("rank=%d j=%d color=%d neighbors colors %d\n",rank,j,min_color,neighbor_colors[m]);
                    }
                  }                  
                  printf("END DEBUG_PARALLEL_COLORING, its processor %d\n", rank);
          #endif
          } // end of if (colors[] == 0)
          /*
          printf("Printing neighbor_colors for vertex %d from processor %d\n", j, rank);
          printArray(V,neighbor_colors) ; */
          free(neighbor_colors);
        } // end of j
        for (j=0; j < range[rank]; j++) {   //for each vertex of this process
          for (k = 0; k < V; k++) {   
            if (p_graph[j * V + k] == 1){
              if (vtx_colors[j] == vtx_colors[j+1]){
                //printf("HELLO ITS ME j %d k %d \n", j, k);
                vtx_colors[j]++; //color
                }}}}

    //each process sends the colors of its vertices to root
    MPI_Gatherv(vtx_colors,range[rank],MPI_INT,colors,range,offsets,MPI_INT,root,MPI_COMM_WORLD);
    //root synchronizes colors on all processes
    MPI_Bcast(colors,V,MPI_INT,root,MPI_COMM_WORLD);
    #if DEBUG_PARALLEL_COLORING
      if (i==1) {
        int p;
        for (p=0;p < range[rank];p++){      
          printf("Checking copy from vtx_colors to colors:rank=%d vertex=%d j_color=%d 
                  colors=%d\n",rank,offsets[rank]+p,vtx_colors[offsets[rank]+p],colors[offsets[rank] + p]);
        }
      }
    #endif
    }

    /*
    printf("Printing vtx_colors processor %d\n", rank);
    printArray(range[rank],vtx_colors) ;
    
    printf("Printing Colors processor %d\n", rank);
    printArray(V,colors) ; */

    free(vtx_colors);
  }



// be able to print for testing purposes
void printArray (int size, int* array){
    int i;
    printf("\n");
    for (i = 0; i < size; i++) {
            printf("%d ", array[i]);
        }
        printf("\n");
    printf("\n\n");
}

// be able to print for testing purposes
void printMatrix (int row, int col, int* matrix){
    int i,j;

    printf("\n");

    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++) {
            printf("%d ", matrix[i * col + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int compare (const void *a, const void *b)
{
  int la = *(const int*) a;
  int lb = *(const int*) b;
  return (la > lb)-(la < lb);
}

//only root will call this function to read the file
//unless DEBUG_VERTEX_DISTRIBUTION is true in which case
//all processes call this function and build their own graph
void read_graph (char *filename)
{
  FILE* file = fopen(filename,"rb");
  // Checking if file is found
  if (!file) {
    printf("Unable to open file %s\n",filename);
    return;
  }

  char line[1000];
  char *token;
  int i, j;
  int max_degree = -1;
  int row_idx, col_idx; //uv edge indices
  
  //read file line by line
  j = 0;
  bool graph_initialized = false;

  while (fgets(line,1000,file) != NULL) {
    
    //read maximum degree of graph
    if (strstr(line,"max degree") != NULL) {
      strtok(line,":");
      token = (char *)strtok(NULL,":");
      max_degree = atoi(token);
      chromaticity_upper = max_degree;
    }
    
    //reading the chromatic number
    if (strstr(line,"chromatic_upper_bound") != NULL) {
      strtok(line,":");
      token = (char *)strtok(NULL,":");
      max_degree = atoi(token);
      chromaticity_upper = max_degree ;
    }

    //tokenize lines
    //printf("Full line %s\n",line);
    strtok(line," ");
    //read number of vertices and edges
    if (strcmp(line,"p")==0) {
      token = (char *)strtok(NULL," ");
      token = (char *)strtok(NULL," ");
      V = atoi(token);
      token = (char *)strtok(NULL," ");
      E = atoi(token);
    }

    //read edges into graph matrix
    // 1.Initialize graph to all zeros
    if (graph_initialized == false && V > 0) {
      adjmatrix = (int *) malloc(V * V * sizeof(int));
      memset(adjmatrix, 0, V * V * sizeof(int));
      graph_initialized = true;
    }
   
    // 2.then load edges into it
    if (strcmp(line,"e") == 0) {
      token = (char *)strtok(NULL," ");
      row_idx = atoi(token)-1; //0 based index
      token = (char *)strtok(NULL," ");
      col_idx = atoi(token)-1; //0 based index
      //
      adjmatrix[row_idx * V + col_idx] = 1;
      adjmatrix[col_idx * V + row_idx] = 1;//symmetric matrix
      j++; //column index
    }
  } //end while
  
  //some files dont have the max_degree. Set it to the max possible chromaticity
  //of a graph which is V if V is odd and V-1 if V is even and would only
  //happen if the graph were complete, i.e. every pair of distinct vertices is
  //connected by a unique edge. Source: Wikipedia page on complete graphs
  if (max_degree == -1) {
    if (V % 2 == 0) 
      {
        chromaticity_upper = V - 1;
      }
    else {
      chromaticity_upper = V;
    }
  } 
  if (rank == root && j != E && j != E/2) {
      printf("Incorrect edge reading: There are %d edges but read %d.\n", E, j);
  }
  fclose(file);
  printf("--Adjacency Matrix from read_graph--\n");
  printMatrix(V,V,adjmatrix);
}

void write_colors(char * filename){
   FILE* file = fopen(filename,"w");
   if (!file) {
    printf("Unable to open file %s\n",filename);
    return;
   }
   int i;   
   for (i = 0; i < V; i++) {
    fprintf(file,"Vertex = %d has color = %d\n", (i+1), colors[i]);
   }
   //write largest color i.e. chromatic number
   qsort(colors, V, sizeof(int), compare);
   fprintf(file,"Largest color was %d\n", colors[V-1]);
   fclose(file);
}
