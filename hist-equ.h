#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

typedef struct{
    int w;
    int h;
    unsigned __int8 * img;
} PGM_IMG;    

typedef struct{
    int w;
    int h;
    unsigned __int8 * img_r;
    unsigned __int8 * img_g;
    unsigned __int8 * img_b;
} PPM_IMG;

typedef struct{
    int w;
    int h;
    unsigned __int8 * img_y;
    unsigned __int8 * img_u;
    unsigned __int8 * img_v;
} YUV_IMG;

typedef struct{
	int* lut_r;
	int* lut_g;
	int* lut_b;
} COLOR_LUT;

typedef struct
{
    int width;
    int height;
    float * h;
    float * s;
    unsigned __int8 * l;
} HSL_IMG;


//read && write && free color image
PPM_IMG read_ppm(const char * path);
void write_ppm(PPM_IMG img, const char * path);
void free_ppm(PPM_IMG img);

//read && write && free gray scale image
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

//convert rgb to hsl && hsl2rgb
HSL_IMG rgb2hsl(PPM_IMG img_in);
PPM_IMG hsl2rgb(HSL_IMG img_in);

//convert rgb to yuv and yuv to rgb
YUV_IMG rgb2yuv(PPM_IMG img_in);
PPM_IMG yuv2rgb(YUV_IMG img_in);    

void histogram(int * hist_out, unsigned __int8  * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned __int8  * img_out, unsigned __int8  * img_in,             //this could will be run in the gpu
                            int * hist_in, int img_size, int nbr_bin);					 //to increase compution and reduce time

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);

//Contrast enhancement for color images
PPM_IMG contrast_enhancement_c_rgb(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_yuv(PPM_IMG img_in);
PPM_IMG contrast_enhancement_c_hsl(PPM_IMG img_in);


#endif
