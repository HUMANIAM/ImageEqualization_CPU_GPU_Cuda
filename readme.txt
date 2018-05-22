In this Task. trying to accelerate computation of image equlization by running the equalization code in the GPU instead of the CPU.
by using the cuda framework to write GPU parallel computing program in the NVIDIA GPU.

steps:
_____
1-read gray scale image and color image

2-apply the serial code of image equalization in the CPU on the gray image and calcuate the time of computation

3-apply the parallel code of image equalization in the GPU on the gray image and calcualte the time of compution

4-apply equalization on every axis (r, g, b) of the color image on the cpu then write it on the disk

5-convert the color image into hsv color system then apply equal on v values on the cpu then write the image after convert it to rgb

6-convert the color image into yuv color system then apply equal on y values on the cpu then write the image after convert it to rgb

7-apply the parallel code of image equalization in the GPU on the color image. equalize every pixel with thread in the GPU

8- show the time that every operation take


* results of applying these step on color image (in-color.jpg) and gray scale image(in-grayscale.jpg) both are 8 bpp

----------------------------------------------------
image 		 |   CPU   (ms)  |  GPU (ms)  
----------------------------------------------------
in-grayscale     |     0.002016  |  10.1332
----------------------------------------------------
in-color         |     0.001952  |  147.497            
____________________________________________________


conclusion
----------
- I try show the difference in exectution time between gpu and cpu with running a sample code that adding 1 millon items of 2 vector and find
   *time : 1266.4(ms)      >> 1 thread running in the gpu to add the 2 vector
   *time : 7.45232(ms)	   >> 256 threads running in the gpu at the same time to add the 2 vector
   *time : 0.001920 (ms)   >> time taken by the cpu

this results are strange either with adding 2 vector or image equalization. this may be because we running in small data and small number of operations
may be we need some optimization.



Resources:
----------

1- http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node2.html
2- https://sites.google.com/site/5kk70gpu/
3- https://github.com/nothings/stb
4- https://devblogs.nvidia.com/even-easier-introduction-cuda/
