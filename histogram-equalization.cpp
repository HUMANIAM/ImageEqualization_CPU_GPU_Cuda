#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned __int8  * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned __int8  * img_out, unsigned __int8  * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);		//create lookup table for mapping from in to out
    int i, min, d;
	float cdf;
	cdf = min = i = 0;
    /* Construct the LUT by calculating the CDF */
  
    while(min == 0) min = hist_in[i++];
    d = img_size - min;

    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        lut[i] = (int)((cdf - min)*255/d + 0.5);

        if(lut[i] < 0)   lut[i] = 0;
		if(lut[i] > 255) lut[i] = 255;
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++)
	   img_out[i] = (unsigned char)lut[img_in[i]];
}



