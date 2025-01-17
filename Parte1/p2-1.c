#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <omp.h>



void calculoMedia(unsigned char **matriz, int filas, int columnas, char *fichero, int hilos){
    FILE *f;
    int suma, i, j, k, l;
    unsigned char maximo;
    unsigned char minimo;
    int iam;
	
	//comenzar a medir el tiempo que se demora
    double inicio = omp_get_wtime();
    
    //asignar espacio para la nueva matriz de calculo de media
    unsigned char **matrizMedia;
	unsigned char **matrizMedia2;
    matrizMedia = (unsigned char**)malloc((filas) * sizeof(unsigned char*));

    #pragma omp parallel num_threads(hilos) private(i, j, k, l) shared(matriz, filas, columnas, matrizMedia) 
	{
		#pragma omp for
		for(i = 0; i<filas; i++){
			matrizMedia[i] = (unsigned char*)malloc((columnas) * sizeof(unsigned char));
		}
	
		//calcular la media de cada dato de la matriz original y guardarlos en la matriz de medias
		#pragma omp for reduction(min:minimo)reduction(max:maximo) reduction(+:suma)
		for (i = 2; i < filas+2; i++){
			
				for (j = 2; j < columnas+2; j++){
					suma = 0;
					maximo = 0;
					minimo = 255;
					
					for (k = -2; k <= 2; k++){
						for (l = -2; l <= 2; l++){
								suma += matriz[i + k][j + l];
								if(matriz[i + k][j + l] > maximo){
									maximo = matriz[i + k][j + l];
								}
								if(matriz[i + k][j + l] < minimo){
									minimo = matriz[i + k][j + l];
								}
						}
					}
					suma = suma - minimo - maximo;

					matrizMedia[i-2][j-2] = suma/ 23;
					
				}
        }
	}
	
	//fin de medición de tiempo
    double fin = omp_get_wtime();
    double tiempo = (fin - inicio)*1000;
    
    printf("\nTiempo medido: %.0f milisegundos.\n", tiempo);
	
	
	//copiar los datos en el fichero binario
	char newFile[25] = "Media";
	strcat(newFile,fichero);
	f = fopen(newFile, "wb");
	for(int i = 0; i<filas; i++){
		fwrite(matrizMedia[i], sizeof(unsigned char), columnas, f);
	}
	fclose(f);
    

    return;
}

void sobel(unsigned char **matriz, int filas, int columnas, char *fichero, int hilos){
    FILE *fp;
    //asignar espacio para la matriz de resultados de C y F, en este caso son ints porque hay resultados que pueden ser negativos
    int cc;
	int iam;
    int ff;
	int bloquesf = filas/(10*hilos)+1;
	int bloquesc = columnas/(10*hilos)+1;
    //declarar las matrices C y F
    int C[3][3] = {{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}};
    int F[3][3] = {{-1, -2, -1},{0, 0, 0},{1, 2, 1}};
	
	//comenzar a medir el tiempo que se demora
    double inicio = omp_get_wtime();
	
    //asignar espacio para la nueva matriz de calculo de media
    unsigned char **matrizSobel;
    matrizSobel = (unsigned char**)malloc((filas) * sizeof(unsigned char*));
	#pragma omp parallel num_threads(hilos) shared(filas, columnas, matriz, matrizSobel, C, F) private(cc, ff)
	{
		#pragma omp for schedule(dynamic,bloquesc)
		for(int i = 0; i<filas; i++){
			matrizSobel[i] = (unsigned char*)malloc((columnas) * sizeof(unsigned char));
		}


		#pragma omp for schedule(dynamic, bloquesf)
		//calcular la detección de bordes (sobel)  de cada dato de la matriz original y guardarlos en la matriz de medias
		for(int i = 3; i<=filas; i++){

			for(int j = 3; j<=columnas; j++){
				cc = 0;
				ff = 0;

				for(int k = -1; k<=1; k++){
					
					for(int l = -1; l<=1; l++){
						cc += matriz[i+k][j+l]*C[k+1][l+1];
						ff += matriz[i+k][j+l]*F[k+1][l+1];
					}
				}	
				matrizSobel[i-2][j-2] = sqrt( cc*cc + ff*ff );
			}
			
		}
	
		//Extensión simétrica 
		#pragma omp for schedule(dynamic,bloquesc)
		for(int i = 1; i<columnas; i++){
			matrizSobel[0][i] = matrizSobel[2][i];
			matrizSobel[filas-1][i] = matrizSobel[filas-3][i];
		}
		#pragma omp for schedule(dynamic,bloquesf)
		for(int i = 1; i<filas; i++){
			matrizSobel[i][0] = matrizSobel[i][2];
			matrizSobel[i][columnas-1] = matrizSobel[i][columnas-3];
		}
	}	
	
	//fin de medición de tiempo
    double fin = omp_get_wtime();
    double tiempo = (fin - inicio)*1000;
    
    printf("\nTiempo medido: %.0f milisegundos.\n", tiempo);
		
		matrizSobel[0][0] = matrizSobel[2][2];
		matrizSobel[0][columnas-1] = matrizSobel[2][columnas-3];
		matrizSobel[filas-1][0] = matrizSobel[filas-3][2];
		matrizSobel[filas-1][columnas-1] = matrizSobel[filas-3][columnas-3];
	
		//copiar los datos en el fichero binario
		char newFile[25] = "Sobel";
		strcat(newFile,fichero);
		fp = fopen(newFile, "wb");

		for(int i = 0; i<filas; i++){
			fwrite(matrizSobel[i], sizeof(unsigned char), columnas, fp);
		}
		fclose(fp);
	
    return;
}   

void histograma(int filas, int columnas, unsigned char **matriz, int hilos){
	FILE *f;

	long contador[256] = {0};
	unsigned char min = 255, max=0;
	int i,j;
	int igualmax = 0;
	int igualmin=0;
	int bloquesf = filas/(hilos*15);
	
	//comenzar a medir el tiempo que se demora
    double inicio = omp_get_wtime();
	
    //contar la cantidad de veces que se repite cada valor en la matriz
	#pragma omp parallel num_threads(hilos) private(i,j ) shared(matriz,contador, filas, columnas)
	{
		
		#pragma omp for reduction(+:contador[:256])
		for(i = 2; i<filas+2; i++){
			for(j = 2; j<columnas+2; j++){
				contador[matriz[i][j]]++;
			}
		}
	}
	
	//calcular maximo y mínimo
	#pragma omp parallel for  private(i,j ) shared(matriz, filas, columnas) reduction(min:min)reduction(max:max)
	for(i = 2; i<filas+2; i++){
		for(j = 2; j<columnas+2; j++){
			if(max<matriz[i][j]){
				max=matriz[i][j];
			}
			if(min>matriz[i][j]){
				min=matriz[i][j];
			}
		}
	}
	
	//calculo de elementos iguales al maximo y al minimo
	/*También se puede hacer accediendo directamente a contador[max] y contador[min]
	pero al pedirlo como región paralela lo he hecho de esta otra manera*/
	#pragma omp parallel for  private(i,j ) shared(matriz, filas, columnas) reduction(+:igualmax, igualmin)
	for(i = 2; i<filas+2; i++){
		for(j = 2; j<columnas+2; j++){
			if(max==matriz[i][j]){
				igualmax++;
			}
			if(min==matriz[i][j]){
				igualmin++;
			}
		}
	}
	
	//fin de medición de tiempo
    double fin = omp_get_wtime();
    double tiempo = (fin - inicio)*1000;
    
    printf("\nTiempo medido: %.0f milisegundos.\n", tiempo);
	
	 //abrir el archivo de texto y copiar los datos del histograma
    f = fopen("histograma.txt","w");
    if (f == NULL) {
        printf("No se puede abrir el archivo\n");
        return;
    }

    for(int i=0; i<256; i++){
        fprintf(f, "%d - %d\n", i, contador[i]);
    }
    fclose(f);
	
    return;
} 

	
void main(int argc, char **argv)
{
	int hilos = atoi(argv[5]);
	
    //double inicio = omp_get_wtime();
    FILE *f;
    unsigned char **matriz;
    int filas = atoi(argv[2]);
    int columnas = atoi(argv[3]);
    int procesado = atoi(argv[4]);
	int i,j;

    // Leer la imagen
    char *fichero = argv[1];

    //Abrimos el archivo binario .raw
    f = fopen(fichero, "rb");
    if (f == NULL) {
        printf("No se puede abrir el archivo binario\n");
        return;
    }

    //Le asignamos espacio de manera dinámica a la matriz en la que guardaremos los datos
    matriz = (unsigned char**)malloc((filas+4) * sizeof(unsigned char*));

    for(i = 0; i<filas+4; i++){
        matriz[i] = (unsigned char*)malloc((columnas+4) * sizeof(unsigned char));
    }

    for(i = 0; i<filas; i++){
        fread(&matriz[i+2][2], sizeof(unsigned char), columnas, f);
    }


    //cerramos el archivo binario
    fclose(f);

    // Extender las filas
		
	for (i = 0; i < 2; i++) {
		for (j = 0; j < columnas; j++) {
			matriz[i][j+2] = matriz[4-i][j+2];
			matriz[filas+2+i][j+2] = matriz[filas-i][j+2];
		}
	}
	//extender las columnas
	for (i = 0; i < filas+4; i++) {
		for (j = 0; j < 2; j++) {
			matriz[i][j] = matriz[i][4-j];
			matriz[i][columnas+2+j] = matriz[i][columnas-j];
		}
	}
	

    
    if(procesado == 1){
		//LLamar a función para hacer el calculo de Media
        calculoMedia(matriz, filas, columnas, fichero, hilos);
    }else if(procesado == 2){
		//LLamar a función para hacer el calculo Sobel
		sobel(matriz, filas, columnas, fichero, hilos);
    }else if(procesado == 3){
		//LLamar a función para hacer el histograma
        histograma(filas, columnas, matriz, hilos);
    }


    return;
}