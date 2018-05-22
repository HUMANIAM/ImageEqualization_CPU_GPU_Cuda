#include <stdio.h>
#include <string.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "hist-equ.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION

//time : 1266.4(ms)
//time : 7.45232(ms)
//time : 0.001920 (ms)

#include "stb-master\\stb_image.h"
#include "stb-master\\stb_image_write.h"
using namespace std;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
const int  NUM_OF_THREADS = 1024; 

void run_cpu_color_test(PPM_IMG);
void run_gpu_color_test(PPM_IMG&, unsigned __int8 *, int , int);
void run_cpu_gray_test(PGM_IMG);
void run_gpu_gray_test(PGM_IMG);
PPM_IMG read_color_image(char*, unsigned __int8 * & );
void free_color_image(PPM_IMG&);
void wirte_color_image(char*, PPM_IMG&);
static void HandleError( cudaError_t, const char *, int);
void calculate_lut(int * , unsigned __int8  * , int , int );
__global__ void histogram_equalization_GPU(unsigned __int8*, unsigned __int8*, int*, int);
__global__ void histogram_equalization__color__GPU(unsigned __int8* ,  unsigned __int8*, COLOR_LUT, int);
//************************************************************************

int main(){
 
	PGM_IMG img_ibuf_g;			   //input gray scale image
	PPM_IMG img_ibuf_c;			   //input color image
	unsigned __int8 * img_c_gpu;   //for gpu
	int n;

	//read gray scale image and color image
	img_ibuf_g.img = stbi_load("inputImages\\in-grayscale.jpg", &(img_ibuf_g.w), &(img_ibuf_g.h), &n, 1);
	img_ibuf_c = read_color_image("inputImages\\in-color.jpg", img_c_gpu);

	cout<<"Start Equalization process with grayscale image\n";
	run_cpu_gray_test(img_ibuf_g);
	run_gpu_gray_test(img_ibuf_g);

	cout<<"\n\nStart Equalization with color image\n";
	run_cpu_color_test(img_ibuf_c);
	run_gpu_color_test(img_ibuf_c, img_c_gpu, img_ibuf_c.w, img_ibuf_c.h);

	//free cpu memory 
	free(img_ibuf_g.img);
	free_color_image(img_ibuf_c);

    return 0;
}

void run_gpu_color_test(PPM_IMG& img_rgb,  unsigned __int8 * img_in, int w, int h)
{
	unsigned __int8 *  out_image;
	unsigned __int8 *  sharedImage;  // gpu can't access cpu memory so copy the image into shared memory
	COLOR_LUT lut;				    //hold lut for r g b axises
	float time;
	cudaEvent_t start, stop;
	const int AR = w * h;
	const int SZ = AR * 3;
    
	cudaMallocManaged(&(sharedImage), SZ * sizeof(unsigned __int8));
	cudaMallocManaged(&(out_image), SZ * sizeof(unsigned __int8));
	//copy the color image in the shared memory
	for(int i=0; i<SZ; i++)
		sharedImage[i] = img_in[i];

	//lut for r axis
	cudaMallocManaged(&(lut.lut_r), 256*sizeof(int));
	calculate_lut(lut.lut_r, img_rgb.img_r, AR, 256);
	//lut for g axis
	cudaMallocManaged(&(lut.lut_g), 256*sizeof(int));
	calculate_lut(lut.lut_g, img_rgb.img_g, AR, 256);
	//lut for b axis
	cudaMallocManaged(&(lut.lut_b), 256*sizeof(int));
	calculate_lut(lut.lut_b, img_rgb.img_b, AR, 256);

	//launch gpu threads to evaluate the image equalization
	const int NUM_OF_BLOCKS = (SZ+NUM_OF_THREADS-1)/NUM_OF_THREADS;
	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );
	histogram_equalization__color__GPU<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(out_image, sharedImage, lut, SZ);
	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	cout<<"Processing time with GPU: "<<time<<"(ms)\n";
    
	stbi_write_png("outimages\\gpu_rgb.png", w, h, 3, out_image, 3*w);
	cudaFree(lut.lut_r);
	cudaFree(lut.lut_g);
	cudaFree(lut.lut_b);
	cudaFree(out_image);
}

void run_gpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG result;
	float time;
	cudaEvent_t start, stop;
	unsigned __int8 *sharedImage;  // gpu can't access cpu memory so copy the image into shared memory

	//evaluate the lut
	const int SZ = img_in.h * img_in.w;
	const int NUM_OF_BLOCKS = (SZ+NUM_OF_THREADS-1)/NUM_OF_THREADS;
	int *lut;
	cudaMallocManaged(&lut, 256*sizeof(int));		//create lookup table for mapping from in to out
	calculate_lut(lut, img_in.img, SZ, 256);

	//equlize the image in the GPU
	result.w = img_in.w;
    result.h = img_in.h;
	cudaMallocManaged(&(result.img), SZ * sizeof(unsigned __int8 ));
	cudaMallocManaged(&sharedImage, SZ*sizeof(unsigned __int8));
 
	// copy from cpu memory to shared memroy
	for (int i = 0; i < SZ; i++)
		sharedImage[i] = img_in.img[i];

	//launch image equalization image in the GPU
	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );
	
	histogram_equalization_GPU<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(result.img, sharedImage, lut, SZ);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	cout<<"Processing time with GPU: "<<time<<"(ms)\n";
    
	stbi_write_png("outimages\\gpu_gray.png", result.w, result.h, 1, result.img, result.w);
	cudaFree(result.img);
}

void run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv, img_obuf_rgb;
	float time;
	cudaEvent_t start, stop;

	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );
    img_obuf_rgb = contrast_enhancement_c_rgb(img_in);
    HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	cout<<"RGB Processing time with CPU: "<<time<<"(ms)\n";

	wirte_color_image("outimages\\cpu_rgb.png", img_obuf_rgb);
	free_color_image(img_obuf_rgb);
    
    HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in);
    HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	cout<<"HSL Processing time with CPU: "<<time<<"(ms)\n";

	wirte_color_image("outimages\\hsv.png", img_obuf_hsl);
	free_color_image(img_obuf_hsl);

    HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in);
    HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	cout<<"YUV Processing time with CPU: "<<time<<"(ms)\n";
    
   wirte_color_image("outimages\\yuv.png", img_obuf_yuv);
   free_color_image(img_obuf_yuv);
}


void run_cpu_gray_test(PGM_IMG img_in)
{
	PGM_IMG img_obuf;
	float time;
	cudaEvent_t start, stop;
	
	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );
	img_obuf = contrast_enhancement_g(img_in);
	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
	printf("Processing time with CPU: %f (ms)\n", time);
    
	stbi_write_png("outimages\\cpu_gray.png", img_obuf.w, img_obuf.h, 1, img_obuf.img, img_obuf.w);
	free(img_obuf.img);
}


PPM_IMG read_color_image(char* path, unsigned __int8 * & img){
	PPM_IMG img_ibuf_c;			//input color image
	int n, cimsz;
	//read color image
	img = stbi_load(path, &(img_ibuf_c.w), &(img_ibuf_c.h), &n, 3);

	cimsz = img_ibuf_c.w * img_ibuf_c.h * n;

	img_ibuf_c.img_r = (unsigned __int8 * ) malloc(img_ibuf_c.w * img_ibuf_c.h*sizeof(unsigned __int8));
	img_ibuf_c.img_g = (unsigned __int8 * ) malloc(img_ibuf_c.w * img_ibuf_c.h*sizeof(unsigned __int8));
	img_ibuf_c.img_b = (unsigned __int8 * ) malloc(img_ibuf_c.w * img_ibuf_c.h*sizeof(unsigned __int8));

	for(int i=0, j=0; i<cimsz; i+=3, j++){
		img_ibuf_c.img_r[j] = img[i];
		img_ibuf_c.img_g[j] = img[i+1];
		img_ibuf_c.img_b[j] = img[i+2];
	}
	return img_ibuf_c;
}

void free_color_image(PPM_IMG& im){
	free(im.img_b);
	free(im.img_g);
	free(im.img_r);
}

void wirte_color_image(char* path, PPM_IMG& img){
	int sz = img.w*img.h;
	unsigned __int8 * outimg = (unsigned __int8 *)malloc(3 * img.w * img.h * sizeof(unsigned __int8));

    for(int i = 0; i < sz; i ++){
        outimg[3*i + 0] = img.img_r[i];
        outimg[3*i + 1] = img.img_g[i];
        outimg[3*i + 2] = img.img_b[i];
    }

	stbi_write_png(path, img.w, img.h, 3, outimg, 3*img.w);
}

static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

void calculate_lut(int * lut, unsigned __int8  * img_in, int img_size, int nbr_bin){
	int i=0, min=0, d=0, cdf=0;

	//evaluate histogram
	int* hist = (int*) malloc(256*sizeof(int));		//commulative distributed funciton
	histogram(hist, img_in, img_size, nbr_bin);

    /* Construct the LUT by calculating the CDF */
	while(min == 0 && i<nbr_bin) min = hist[i++];
    d = img_size - min;

	//construct the lookup table
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);

        if(lut[i] < 0)   lut[i] = 0;
		if(lut[i] > 255) lut[i] = 255;
    }

	//for(int i=0; i<nbr_bin; i++) cout<<i<<"   "<<lut[i]<<endl;
}

__global__ void histogram_equalization_GPU(unsigned __int8* img_out, unsigned __int8* img_in, int* lut, int img_size){
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	if(x<img_size)
		img_out[x] = lut[img_in[x]];
}

__global__ void histogram_equalization__color__GPU(unsigned __int8* img_out,  unsigned __int8* img_in, COLOR_LUT lut, int sz){
	int x = threadIdx.x + blockDim.x * blockIdx.x; 
	if(x<sz){
		if(x%3 == 0)
			img_out[x] = lut.lut_r[img_in[x]];
		else if(x%3 == 1)
			img_out[x] = lut.lut_g[img_in[x]];
		else 
			img_out[x] = lut.lut_b[img_in[x]];
	}
	
}