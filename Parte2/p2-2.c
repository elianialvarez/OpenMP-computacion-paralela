#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>

int numerico(char **argv){
    for(int j=1; j<4; j++){
        int i=0;
        while (argv[j][i] != '\0') {
            if (!isdigit(argv[j][i])) {
                return 1;
            }
            i++;
        }
    }
    return 0;
}

void main(int argc, char *argv[]){

    int m;
    int N;
    unsigned char sin_fichero=1;
    double **matriz;
    double *x;
    double *x_aux;
    char *fichero;
    FILE *f;
    double euclides;
    double normalizado;
    double suma;


    if(argc != 4 && argc != 5){
        printf("Es necesario poner como argumento:\n 1- numero de iteraciones(numero entero)\n 2- tamano de la matriz(numero entero)\n 4- fichero\n");
    }
    if(numerico(argv) == 1){
        printf("Alguno de los valores que deberia ser numerico no lo es");
        return;
    }

    m=atoi(argv[1]);
    N=atoi(argv[2]);
	int hilos = atoi(argv[3]);
	

    //asignarle espacio a la matriz original
    matriz = (double**)malloc(sizeof(double*)*N);

	//asignarle espacio al vector x y x_aux;
	x = (double*)malloc(sizeof(double)*N);
	x_aux = (double*)malloc(sizeof(double)*N);
	
	
	//asignarle espacio a las filas de matriz original e igualar cada valor de x a 1.0
	for(int i = 0; i < N; i++){
		matriz[i] = (double*)malloc(sizeof(double)*N);
		x[i]=1.0;
	}

	
    //pedir los datos del tamaño de la matriz, el nombre del fichero y el número de iteraciones al usuario
    if(argc==5){
        fichero=argv[4];
        //si el archivo no existe se crea la matriz aleatoriamente
        f=fopen(fichero,"rb");
        if (f == NULL) {
            printf("El archivo no existe, se generara una matriz aleatoriamente\n");
            sin_fichero=1;
        }else{
            fseek(f, 0, SEEK_END);
            long tam_fichero = ftell(f)/sizeof(double);
            //si el fichero seleccionado tiene menos datos de los que necesita la matriz, esta se crea aleatoriamente
            if(tam_fichero < N*N){
                printf("No hay suficientes datos en el archivo seleccionado para generar la matriz, se generara una matriz aleatoriamente\n");
                sin_fichero=1;
            }else{
                //se copian los datos del fichero en la matriz
                sin_fichero=0;
                fseek(f, 0, SEEK_SET);
                for(int i = 0; i < N; i++){
                    fread(matriz[i],sizeof(double),N,f);
                }
            }
            fclose(f);
        }
    }
	
	//comenzar a medir el tiempo que se demora
    double inicio = omp_get_wtime();

    //se genera la matriz aleatoriamente si no hay fichero viable
    if(sin_fichero==1){
		#pragma omp parallel num_threads(hilos) shared(matriz, N)
        {
			unsigned int seed = omp_get_thread_num() + 1;
			#pragma omp for		
			for(int i = 0; i < N; i++){
				for(int j = 0; j < N; j++){
					if(i==j){
						matriz[i][j] = 1;
					}else{
						matriz[i][j] =  ((double)rand_r(&seed) / (double)RAND_MAX) * 0.02 - 0.01;
					}
				}
			}
		}
    }
    //iteraciones para realizar los calculos necesarios
    for(int i=0; i<m; i++){
		#pragma omp parallel num_threads(hilos) shared(matriz, N, i, normalizado, x_aux, x) private(suma)
        {
			//se multiplica el vector por la matriz y se le resta el resultado al vector resultante de la iteración anterior
			#pragma omp for
			for(int k = 0; k < N; k++){
				suma=0;
				for(int j = 0; j < N; j++){
					suma+=matriz[k][j]*x[j];
					
				}
				
				if(i==0){
					x_aux[k] = suma;
				}
				else{
					x_aux[k] = x[k]-suma;
				}
			}

			//realizamos euclides
			euclides=0;
			#pragma omp for reduction(+:euclides)
			for(int j=0; j<N; j++){
				euclides+=pow(x_aux[j],2);
			}
			
			#pragma omp single
			{
				normalizado=sqrt(euclides);
				printf("Norma de iteracion %d:\n",i+1);
				printf("%.6e\n", normalizado);
			}
			
			//terminamos de normalizar el vector
			#pragma omp for
			for(int j=0; j<N; j++){
				x[j]=x_aux[j]/normalizado;
			}
		}
        
    }  

	//fin de medición de tiempo
    double fin = omp_get_wtime();
    double tiempo = (fin - inicio);
    
    printf("\nTiempo medido: %.2f segundos.\n", tiempo);	

    free(x_aux);
    free(x);
    for(int j=0; j<N; j++){
        free(matriz[j]);
    }
    free(matriz);


    return;
}