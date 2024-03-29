#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <math.h>

#ifndef TOPEFFT_H
#define TOPEFFT_H
#include "topefft.h"
#endif

void Xtope1DExecMix( struct topeFFT *f,
					 struct XtopePlan1D *t,
					 int type)
{
	f->error = clSetKernelArg(t->kernel[type], 5, sizeof(int), (void*)&type);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel_swap, 4, sizeof(int), (void*)&type);
	clSetKernelArgChecker(&f->error);
	$CHECKERROR
    
	/* Run Swapper */
	switch(type)
	{
		case 0:		t->globalSize[0] = t->side[0];
					t->globalSize[1] = t->side[1];
					t->localSize[0] = 1;//t->radix[0] < 64 ? t->radix[0]/2 : 64;
					t->localSize[1] = 1;
					break;
		case 1: 	t->globalSize[0] = t->side[0];
					t->globalSize[1] = t->side[1];
					t->localSize[0] = 1;
					t->localSize[1] = 1;//t->radix[1] < 64 ? t->radix[1]/2 : 64;
					break;
	}

	f->error = 
		clSetKernelArg(t->kernel_swap,3,sizeof(cl_mem), (void*)&t->bitrev[type]);
	$CHECKERROR

	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_swap,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
	
	#if 0 // Debug Code
	int i, j;
	double *d = malloc(sizeof(double)*2*t->length);
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
								CL_TRUE, 0, t->dataSize, d, 
								0, NULL, &f->event);
	$CHECKERROR
	for (i = 0; i < t->side[0]; i++) {
		for (j = 0; j < t->side[1]; j++) {
			printf("%.3lf:%.3lf\t", d[2*(j*t->side[0]+i)], 
									d[2*(j*t->side[0]+i)+1]);
		}
		printf("\n");
	}
	exit(0);
	#endif

	/* Run Butterflies */
	switch(type)
	{
		case 0:
			t->globalSize[1] = t->side[1];
			t->localSize[1] = 1;
			switch(t->radix[0])
			{
				case 10:t->globalSize[0] = t->side[0]/10;
						t->localSize[0] = t->side[0]/10 < 100 ? t->side[0]/10 : 100;
						break;
				case 8: t->globalSize[0] = t->side[0]/8;
						t->localSize[0] = t->side[0]/8 < 64 ? t->side[0]/8 : 64;
						break;
				case 7: t->globalSize[0] = t->side[0]/7;
						t->localSize[0] = t->side[0]/7 < 49 ? t->side[0]/7 : 49;
						break;
				case 6: t->globalSize[0] = t->side[0]/6;
						t->localSize[0] = t->side[0]/6 < 36 ? t->side[0]/6 : 36;
						break;
				case 5: t->globalSize[0] = t->side[0]/5;
						t->localSize[0] = t->side[0]/5 < 125 ? t->side[0]/5 : 125;
						break;
				case 4: t->globalSize[0] = t->side[0]/4;
						t->localSize[0] = t->side[0]/4 < 128 ? t->side[0]/4 : 128;
						break;
				case 3: t->globalSize[0] = t->side[0]/3;
						t->localSize[0] = t->side[0]/3 < 81 ? t->side[0]/3 : 81;
						break;
				case 2: t->globalSize[0] = t->side[0]/2;
						t->localSize[0] = t->side[0]/2 < 128 ? t->side[0]/2 : 128;
						break;
			}
			break;
		case 1:
			t->globalSize[0] = t->side[0];
			t->localSize[0] = 1;
			switch(t->radix[1])
			{
				case 10:t->globalSize[1] = t->side[1]/10;
						t->localSize[1] = t->side[1]/10 < 100 ? 1 : 100;
						break;
				case 8: t->globalSize[1] = t->side[1]/8;
						t->localSize[1] = t->side[1]/8 < 64 ? t->side[1]/8 : 64;
						break;
				case 7: t->globalSize[1] = t->side[1]/7;
						t->localSize[1] = t->side[1]/7 < 49 ? t->side[1]/7 : 49;
						break;
				case 6: t->globalSize[1] = t->side[1]/6;
						t->localSize[1] = t->side[1]/6 < 36 ? t->side[1]/6 : 36;
						break;
				case 5: t->globalSize[1] = t->side[1]/5;
						t->localSize[1] = t->side[1]/5 < 125 ? t->side[1]/5 : 125;
						break;
				case 4: t->globalSize[1] = t->side[1]/4;
						t->localSize[1] = t->side[1]/4 < 128 ? t->side[1]/4 : 128;
						break;
				case 3: t->globalSize[1] = t->side[1]/3;
						t->localSize[1] = t->side[1]/3 < 81 ? t->side[1]/3 : 81;
						break;
				case 2: t->globalSize[1] = t->side[1]/2;
						t->localSize[1] = t->side[1]/2 < 128 ? t->side[1]/2 : 128;
						break;
			}
			break;
	}

	#if 1 // Debug Code
	printf("ND_RangeX: %d WG_X: %d\n", t->globalSize[0], t->localSize[0]);
	printf("ND_RangeY: %d WG_Y: %d\n", t->globalSize[1], t->localSize[1]);
	printf("Radix %c: %d\n", type == 0 ? 'X' : 'Y', t->radix[type]);
	printf("Stages: %d\n", t->log[type]);
	#endif

	int s;
	for (s = 1; s <= t->log[type]; s++) {
		f->error = clSetKernelArg(t->kernel[type], 4, sizeof(int), (void*)&s);
		$CHECKERROR

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel[type],
											t->dim, NULL, t->globalSize,
											t->localSize, 0, NULL, &f->event);
		clEnqueueNDRangeKernelChecker(&f->error);
		$CHECKERROR
		clFinish(f->command_queue);
		t->totalKernel += profileThis(f->event);

		#if 0 // Debug Code
		int i;
		double *d = malloc(sizeof(double)*2*t->length);
		f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
		$CHECKERROR
		for (i = 0; i < t->length; i++) {
			printf("%lf:%lf\n", d[2*i], d[2*i+1]);
		}
		exit(0);
		#endif
	}

	#if 0 // Debug Code
	int i, j;
	double *d = malloc(sizeof(double)*2*t->length);
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
								CL_TRUE, 0, t->dataSize, d, 
								0, NULL, &f->event);
	$CHECKERROR
	for (i = 0; i < t->side[0]; i++) {
		for (j = 0; j < t->side[1]; j++) {
			printf("%.3lf:%.3lf\t", d[2*(j*t->side[0]+i)], 
			d[2*(j*t->side[0]+i)+1]);
		}
		printf("\n");
	}
	exit(0);
	#endif
}

void Xtope1DExecPlain( struct topeFFT *f,
					   struct XtopePlan1D *t)
{
	/* Run Swapper */	
	f->error = 
		clSetKernelArg( t->kernel_swap, 0, sizeof(cl_mem), 
						(void*)&t->data); 
	$CHECKERROR
	
	/* For Swap */
	t->globalSize[0] = t->length;
	switch(t->radix[0])
	{
		case 8: 
		case 4:
		case 2: t->localSize[0] = t->length < 128 ? t->length/2 : 128;
				break;
		case 3: t->localSize[0] = t->length < 243 ? t->length/3 : 243;
				break;
		case 5: t->localSize[0] = t->length < 125 ? t->length/5 : 125;
				break;
		case 6: t->localSize[0] = t->length < 36 ? t->length/6 : 36;
				break;
		case 7: t->localSize[0] = t->length < 49 ? t->length/7 : 49;
				break;
		case 10:t->localSize[0] = t->length < 100 ? t->length/10 : 100;
				break;
	}
	
	f->error = 
		clEnqueueNDRangeKernel(	f->command_queue, t->kernel_swap,
								t->dim, NULL, t->globalSize,
								t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);

	/* Run Butterflies */
	int stage_arg;
	switch(t->radix[0])
	{
		case 10:t->globalSize[0] = (t->length)/10;
				t->localSize[0] = ((t->length)/10) < 100 ? (t->length)/10 : 100;
				stage_arg = 2;
				break;
		case 8: t->globalSize[0] = (t->length)/8;
				t->localSize[0] = ((t->length)/8) < 128 ? (t->length)/8 : 128;
				stage_arg = 3;
				break;
		case 7: t->globalSize[0] = t->length/7;
				t->localSize[0] = t->length/7 < 49 ? t->length/7 : 49;
				stage_arg = 2; 
				break;
		case 6: t->globalSize[0] = t->length/6;
				t->localSize[0] = t->length/6 < 36 ? t->length/6 : 36;
				stage_arg = 2;
				break;
		
		case 5: t->globalSize[0] = t->length/5;
				t->localSize[0] = t->length/5 < 125 ? t->length/5 : 125;
				stage_arg = 2;
				break;
		case 4: t->globalSize[0] = (t->length)/4;
				t->localSize[0] = ((t->length)/4) < 128 ? (t->length)/4 : 128;
				stage_arg = 3;
				break;
		case 3: t->globalSize[0] = t->length/3;
				t->localSize[0] = t->length/3 < 243 ? 1 : 243;
				stage_arg = 2;
				break;
		case 2: t->globalSize[0] = (t->length)/2;
				t->localSize[0] = ((t->length)/2) < 128 ? (t->length)/2 : 128;
				stage_arg = 3;
				break;
	}
	
	int s;
	//setting RC2C radix argument
	clSetKernelArg(t->kernel[0],4,sizeof(int),(void*)&t->radix[0]);  
	for (s = 1; s <= t->log[0]; s++) {
		f->error = 
			clSetKernelArg(t->kernel[0], stage_arg, sizeof(int), (void*)&s);
		$CHECKERROR

		f->error = 
			clEnqueueNDRangeKernel(	f->command_queue, t->kernel[0],
									t->dim, NULL, t->globalSize,
									t->localSize, 0, NULL, &f->event);
		$CHECKERROR
		clFinish(f->command_queue);
		t->totalKernel += profileThis(f->event);
	}
	#if 0 // Debug Code
	int i;
	double *d = malloc(sizeof(double)*2*t->length);
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
									CL_TRUE, 0, t->dataSize, d, 
									0, NULL, &f->event);
	$CHECKERROR
	for (i = 0; i < t->length; i++) {
		printf("%d %lf:%lf\n", i, d[2*i], d[2*i+1]);
	}
	printf("---\n");
	exit(0);
	#endif
}

void Xtope1DExecDFT(struct topeFFT *f, struct XtopePlan1D *t, int type)
{
	f->error = 
		clSetKernelArg(t->kernel[type], 4, sizeof(int), (void*)&type);
	$CHECKERROR

	t->globalSize[0] = t->side[0];
	t->localSize[0] = 1; 
//		t->side[0] < 512 ? t->side[0] : t->side[0] % 2 == 0 ? 2 : 1;

	if (t->dim == 2) {
		t->globalSize[1] = t->side[1];
		t->localSize[1] = 1;
//			t->side[1] < 512 ? t->side[1] : t->side[1] % 2 == 0 ? 2 : 1;
	}

	f->error = 
		clEnqueueNDRangeKernel(	f->command_queue, t->kernel[type],
							t->dim, NULL, t->globalSize,
							t->localSize, 0, NULL, &f->event);
	clEnqueueNDRangeKernelChecker(&f->error);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalKernel += profileThis(f->event);
	
	// copy from scratch to data
	f->error = 
		clEnqueueNDRangeKernel(	f->command_queue, t->kernel_copy,
							t->dim, NULL, t->globalSize,
							t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);
}

void Xtope1DExec(	struct topeFFT *f,
					struct XtopePlan1D *t, 
					double *d, int dir) 
{
	double re;
	int dir_arg;

	dir_arg = 3; // DFT and Any Mix
	if (t->dim == 1) {
		if (modf(log2(t->length),&re) == 0) dir_arg = 4; // radix 2 types
		else 								dir_arg = 3; // radix n types
	} 

	/* Set Direction of Transform */
	int l;
	for (l = 0; l < t->dim; l++) {
		f->error = clSetKernelArg( 	t->kernel[l], dir_arg, 
									sizeof(int), (void*)&dir);
		$CHECKERROR
	}

	if (t->dim == 1) {
		if (t->radix[0] > 1)
			Xtope1DExecPlain(f,t);
		else 
			Xtope1DExecDFT(f,t,0);
	}
	else if (t->dim == 2) {
		/* Run FFT along Y */
		if (t->radix[1] == -1 ) {
			Xtope1DExecDFT(f,t,1);
		}
		else {
			Xtope1DExecMix(f,t,1); // Note: This must be 1
		}

		#if 1
		/* Middle Step */
		t->globalSize[0] = t->side[0];
		t->globalSize[1] = t->side[1];
		t->localSize[0]  = 1;
		t->localSize[1]  = 1;
		
		f->error = 
			clEnqueueNDRangeKernel ( f->command_queue, t->kernel_mulTW,
									 t->dim, NULL, t->globalSize,
									 t->localSize, 0, NULL, &f->event);
		$CHECKERROR
		clFinish(f->command_queue);
		t->totalKernel += profileThis(f->event);

		/* Run FFT along X */
		if (t->radix[0] == -1 ) {
			Xtope1DExecDFT(f,t,0);
		}
		else {
			Xtope1DExecMix(f,t,0); // Note: This must be 1
		}

		/* Perform a Transpose */
		t->globalSize[0] = t->side[0];
		t->globalSize[1] = t->side[1];
		t->localSize[0]  = 1;
		t->localSize[1]  = 1;

		f->error = 
			clEnqueueNDRangeKernel( f->command_queue, t->kernel_tran2,
									t->dim, NULL, t->globalSize,
									t->localSize, 0, NULL, &f->event);
		$CHECKERROR
		clFinish(f->command_queue);
		t->totalKernel += profileThis(f->event);
		#endif
	}

	#if 0 // Debug Code
	int i, j;
	f->error = 
		clEnqueueReadBuffer(	f->command_queue, t->data,
								CL_TRUE, 0, t->dataSize, d, 
								0, NULL, &f->event);
	$CHECKERROR
	
	for (j = 0; j < t->side[1]; j++) {
		for (i = 0; i < t->side[0]; i++) {
			printf("%f+%fi\t", 
				d[2*(j*t->side[0]+i)], d[2*(j*t->side[0]+i)+1]);
		}
		printf("\n");
	}
	exit(0);
	#endif

	/* Divide by N if INVERSE */
	if (dir == 0) {
		t->globalSize[0] = t->length;
		t->localSize[0] = t->length < 512 ? t->length/2 : 256;

		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_div,
											t->dim, NULL, t->globalSize,
											t->localSize, 0, NULL, 
											&f->event);
		$CHECKERROR
	}

	/* Read Data Again */
	f->error = clEnqueueReadBuffer(	f->command_queue, t->data,
								CL_TRUE, 0, t->dataSize, d, 
								0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);
}

void Xtope1dPlanInitMix(	struct topeFFT *f,
							struct XtopePlan1D *t)
{
	/* Kernel Setup 
	 * */
	t->kernel = malloc(sizeof(cl_kernel)*t->dim);
	int ii;
	for (ii = 0; ii < t->dim; ii++) {
		#if 1
		switch(t->radix[ii]) 
		{
			case 2:	t->kernel[ii] = clCreateKernel(	f->program1D, 
													"DIT2C2CM", &f->error);
					$CHECKERROR break;
			case 4:	t->kernel[ii] = clCreateKernel(	f->program1D,
													"DIT4C2CM", &f->error);
					$CHECKERROR break;
			case 8:	t->kernel[ii] = clCreateKernel(	f->program1D,
													"DIT8C2CM", &f->error);
					$CHECKERROR break;
			case 3:
			case 5:
			case 6:
			case 7:
			case 10:t->kernel[ii] = clCreateKernel( f->program1D,
													"DITRC2CM", &f->error);
					$CHECKERROR 
					f->error = clSetKernelArg(	t->kernel[ii], 6, sizeof(int), 
									(void*)&t->radix[ii]);
					$CHECKERROR
					break;
			case -1:t->kernel[ii] = 
						clCreateKernel( f->program1D, "DFTM", &f->error);
					$CHECKERROR
					t->scratch = clCreateBuffer(f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*t->length,
								NULL, &f->error); 			
					$CHECKERROR
					f->error = clSetKernelArg(	t->kernel[ii], 5, sizeof(cl_mem), 
									(void*)&t->scratch);
					$CHECKERROR
					t->kernel_copy = 
						clCreateKernel(f->program1D, "scratchToData", &f->error);
					$CHECKERROR
					f->error = 
						clSetKernelArg( t->kernel_copy, 0, 
										sizeof(cl_mem), (void*)&t->data);
					$CHECKERROR
					f->error = 
						clSetKernelArg(t->kernel_copy, 1, 
										sizeof(cl_mem), (void*)&t->scratch);
					$CHECKERROR
					break;
		}
		f->error = clSetKernelArg(	t->kernel[ii], 0, sizeof(cl_mem), 
									(void*)&t->data);
		$CHECKERROR
		f->error = clSetKernelArg(	t->kernel[ii], 1, sizeof(int), 
									(void*)&t->side[0]);
		$CHECKERROR
		f->error = clSetKernelArg(	t->kernel[ii], 2, sizeof(int), 
									(void*)&t->side[1]);
		$CHECKERROR
		#endif
	}

	/* Transpose Kernel */
	t->kernel_tran2 = clCreateKernel (f->program1D, "transpose2", &f->error);
	$CHECKERROR
	f->error = clSetKernelArg( 	t->kernel_tran2, 0, 
								sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	for (ii = 0; ii < t->dim; ii++) {
		f->error = clSetKernelArg( 	t->kernel_tran2, ii+1, 
		 							sizeof(int), (void*)&t->side[ii]);
		$CHECKERROR
	}

	/* Multiplication Kernel */
	t->kernel_mulTW = clCreateKernel( f->program1D, "kernelMUL", &f->error);
		$CHECKERROR
	f->error = 
		clSetKernelArg(t->kernel_mulTW, 0, sizeof(cl_mem), (void*)&t->data);
		$CHECKERROR
	f->error = 
		clSetKernelArg(t->kernel_mulTW, 1, sizeof(int), (void*)&t->side[0]);
		$CHECKERROR
	f->error = 
		clSetKernelArg(t->kernel_mulTW, 2, sizeof(int), (void*)&t->side[1]);
		$CHECKERROR
	
	/* Bit Reversal */
	t->kernel_bit = malloc(sizeof(cl_kernel)*t->dim);
	t->kernel_bit[0] = clCreateKernel(	f->program1D, "reverse2", &f->error);
	$CHECKERROR
	t->kernel_bit[1] = clCreateKernel(	f->program1D, "reversen", &f->error);
	$CHECKERROR

	for (ii = 0; ii < t->dim; ii++) {
		switch(t->radix[ii])
		{
			case 8:
			case 4:
			case 2: f->error = 
						clSetKernelArg( t->kernel_bit[0], 0, sizeof(cl_mem), 
										(void*)&t->bitrev[ii]); $CHECKERROR
					f->error =
						clSetKernelArg(	t->kernel_bit[0],1,sizeof(int), 
										(void*)&t->bits[ii]);	$CHECKERROR
					break;
			case 10:
			case 7:
			case 6:
			case 5:
			case 3: f->error = 
						clSetKernelArg(	t->kernel_bit[1], 0, sizeof(cl_mem), 
										(void*)&t->bitrev[ii]); $CHECKERROR
					f->error = 
						clSetKernelArg(	t->kernel_bit[1], 1, 
										sizeof(int)*t->bits[ii]*t->side[ii], 
										NULL); 					$CHECKERROR
					f->error = 
						clSetKernelArg(	t->kernel_bit[1], 2, sizeof(int), 
										(void*)&t->bits[ii]);    $CHECKERROR
					f->error = 
						clSetKernelArg(	t->kernel_bit[1], 3, sizeof(int), 
										(void*)&t->radix[ii]);  $CHECKERROR	
					break;
		}
		
		t->globalSize[0] = t->side[ii];
		switch(t->radix[ii])
		{
			case 10: t->localSize[0] = t->side[ii] < 100 ? t->side[ii]/10 : 100;
					 break;
			case 8:  t->globalSize[0] = t->side[ii]/2;
					 t->localSize[0] = t->side[ii]/8 < 64 ? t->side[ii]/8 : 64;
					 break;
			case 7:  t->localSize[0] = t->side[ii]/7 < 49 ? t->side[ii]/7 : 49;
					 break;
			case 6:  t->localSize[0] = t->side[ii]/6 < 36 ? t->side[ii]/6 : 36;
					 break;
			case 5:  t->localSize[0] = t->side[ii]/5 < 125 ? t->side[ii]/5 : 125;
					 break;
			case 4:  t->globalSize[0] = t->side[ii]/2;
					 t->localSize[0] = t->side[ii]/4 < 256 ? t->side[ii]/4 : 256;
					 break;
			case 3:  t->localSize[0] = t->side[ii]/3 < 243 ? t->side[ii]/3 : 243;
					 break;
			case 2:  t->globalSize[0] = t->side[ii]/2;
					 t->localSize[0] = t->side[ii]/2 < 256 ? t->side[ii]/2 : 256;
					 break;
		}
		
		switch(t->radix[ii])
		{
			case 10:
			case 7:
			case 6:
			case 5:
			case 3: f->error = clEnqueueNDRangeKernel(	
						f->command_queue, t->kernel_bit[1],	1, NULL, // t->dim=1
						t->globalSize, t->localSize, 0, NULL, &f->event);
					$CHECKERROR
					break;
			case 8:
			case 4:
			case 2: f->error = clEnqueueNDRangeKernel(	
						f->command_queue, t->kernel_bit[0],	1, NULL, // t->dim=1
						t->globalSize, t->localSize, 0, NULL, &f->event);
					$CHECKERROR
					break;
		}
		clFinish(f->command_queue);
		t->totalPreKernel += profileThis(f->event);
		
		#if 0 // Debug Code
		int *bit = malloc(sizeof(int)*t->side[ii]);
		f->error = 
			clEnqueueReadBuffer( f->command_queue, t->bitrev[ii],
								 CL_TRUE, 0, sizeof(int)*t->side[ii], bit,
								 0, NULL, NULL);
			$CHECKERROR
		int iii;
		for (iii = 0; iii < t->side[ii]; iii++) {
			printf("%d\n", bit[iii]);	
		}
		printf("----\n");
		#endif
	}
}


void Xtope1DPlanInitDFT( 	struct topeFFT *f,
							struct XtopePlan1D *t)
{
	/* Twiddle Setup 
	 * -------------
	 * Dev: Note:
	 * ---------
	 * The size of the twiddle input is +1'd to allow calculation of the last 
	 * twiddle. Eventually, the twiddle calculations would be /4 and recovered 
	 * using the quad/buad technique from the radix kernels in the code.
	 * For example, the last twiddle for input size 5 would be W^16.
	 *
	 * Also, if twiddles are calculated on the fly, the twiddle code in this
	 * section is redundant.
	 */

	#if 0
	int required = (x-1) * (x-1) +1; // Eventually reduce/4 !!!!!

	t->twiddle = clCreateBuffer(f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*required,
								NULL, &f->error);
	$CHECKERROR

	t->kernel_twid = clCreateKernel(f->program1D, "twid1D", &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(	t->kernel_twid, 0, sizeof(cl_mem), 
								(void*)&t->twiddle);	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_twid, 1, sizeof(int), 
								(void*)&t->x);		$CHECKERROR

	t->globalSize[0] = 512;//required;
	if (required % 2 == 0) {
		t->localSize[0] = 32;//required < 512 ? required : 2; 
	}
	else {
		t->localSize[0] = 1;//required < 512 ? required : 1; 
	}
	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_twid,
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
	#endif

	#if 0 // Debug Code
	int i,j;
	double d[required*2];
	f->error = clEnqueueReadBuffer(	f->command_queue, t->twiddle,
									CL_TRUE, 0, sizeof(double)*2*required, d, 
									0, NULL, &f->event);
	for (i = 0; i < t->x; i++) {
		for (j = 0; j < t->x; j++) {
			printf("%f\t%f\t%d\n", d[(2*i*j)], d[(2*i*j)+1], i*j);	
		}
	}
	//exit(0);
	#endif

	/* Create Scratch Space
	 * ---------------------
	 *  This space is required because OpenCL is giving problems in long roops
	 *  and therefore in order to break up the loops into chunks, we need to
	 *  store the multiplye-and-adds in temporary storge (so the original data
	 *  is not touched)
	 *
	 *  Update: It had nothing to do with long loops !!!! hahaha
	 */
	t->scratch = clCreateBuffer(f->context, CL_MEM_READ_WRITE,
								sizeof(double)*2*t->length,
								NULL, &f->error); 			$CHECKERROR

	/* Kernel Setup */
	t->kernel = malloc(sizeof(cl_kernel));
	t->kernel[0] = clCreateKernel(f->program1D, "DFTM", &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(t->kernel[0], 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel[0], 1, sizeof(int), (void*)&t->side[0]);
	$CHECKERROR
	int dummy = 1;
	f->error = clSetKernelArg(t->kernel[0], 2, sizeof(int), (void*)&dummy);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel[0], 5, sizeof(cl_mem), (void*)&t->scratch);
	$CHECKERROR

	/* Scratch to Data Kernel */
	t->kernel_copy = clCreateKernel(f->program1D, "scratchToData", &f->error);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel_copy, 0, sizeof(cl_mem), (void*)&t->data);
	$CHECKERROR
	f->error = clSetKernelArg(t->kernel_copy, 1, sizeof(cl_mem), (void*)&t->scratch);
	$CHECKERROR
}

void Xtope1DPlanInitBaseN( 	struct topeFFT *f, struct XtopePlan1D *t)
{
	/* Kernel Setup
	 */
	t->kernel = malloc(sizeof(cl_kernel));
	switch(t->radix[0])
	{
		case 3:
		case 5:
		case 6:
		case 7:
		case 10:t->kernel[0] = 
					clCreateKernel(f->program1D, "DITRC2C", &f->error);
				$CHECKERROR
				break;
	}
	f->error = 
		clSetKernelArg(t->kernel[0], 0, sizeof(cl_mem), (void*)&t->data);
		$CHECKERROR
	f->error = 
		clSetKernelArg(t->kernel[0], 1, sizeof(int), (void*)&t->length);
		$CHECKERROR

	/* Bit Reversal */
	t->kernel_bit = malloc(sizeof(cl_kernel));
	t->kernel_bit[0] = clCreateKernel(	f->program1D, "reversen", &f->error);
	$CHECKERROR

	f->error = clSetKernelArg(	t->kernel_bit[0], 0, sizeof(cl_mem), 
								(void*)&t->bitrev[0]); 
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit[0], 1, 
								sizeof(int)*t->log[0]*t->radix[0], NULL); 
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit[0], 2, sizeof(int), 
								(void*)&t->log[0]);    
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_bit[0], 3, sizeof(int), 
								(void*)&t->radix[0]);  
	$CHECKERROR	

	t->globalSize[0] = t->length;
	t->localSize[0] = t->radix[0];

	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_bit[0],
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
	
	#if 0 // Debug Code
	int *bit = malloc(sizeof(int)*t->length);
	f->error = 
		clEnqueueReadBuffer( f->command_queue, t->bitrev[0],
							 CL_TRUE, 0, sizeof(int)*t->length, bit,
							 0, NULL, NULL);
	$CHECKERROR
	int iii;
	for (iii = 0; iii < t->length; iii++) {
		printf("%d\n", bit[iii]);	
	}
	printf("----\n");
	exit(0);
	#endif

}

void Xtope1DPlanInitBase2(	struct topeFFT *f,
							struct XtopePlan1D *t)
{
	/* Kernel Setup 
	 **/
	t->kernel = malloc(sizeof(cl_kernel));
	switch(t->radix[0])
	{
		case 2:	t->kernel[0] = 
					clCreateKernel(f->program1D, "DIT2C2C", &f->error);
					$CHECKERROR
				break;
		case 4:	t->kernel[0] = 
					clCreateKernel(f->program1D, "DIT4C2C", &f->error);
					$CHECKERROR
				break;
		case 8:	t->kernel[0] = 
					clCreateKernel(f->program1D, "DIT8C2C", &f->error);
					$CHECKERROR
				break;
	}

	/* Twiddle Setup 
	 **/
	t->twiddle = clCreateBuffer( f->context, CL_MEM_READ_WRITE,
						sizeof(double)*2*t->length/4, NULL, &f->error);
	$CHECKERROR

	int use; // use pre-computed Twiddles or not?
	if (t->side[0] > t->radix[0]) 	use = 1;
	else 							use = 0;
	
	f->error = clSetKernelArg( t->kernel[0], 5, sizeof(int), (void*)&use);
	$CHECKERROR

	if (use == 1) {
		t->kernel_twid = clCreateKernel(f->program1D, "twid1D", &f->error);
		$CHECKERROR

		f->error = clSetKernelArg(	t->kernel_twid, 0, sizeof(cl_mem), 
									(void*)&t->twiddle);	$CHECKERROR
		f->error = clSetKernelArg(	t->kernel_twid, 1, sizeof(int), 
									(void*)&t->length);		$CHECKERROR

		t->globalSize[0] = t->length/4;
		t->localSize[0] = t->length/4 < 512 ? t->length/4 : 256/4;
		f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_twid,
											t->dim, NULL, t->globalSize,
											t->localSize, 0, NULL, &f->event);
		$CHECKERROR
		clFinish(f->command_queue);
		t->totalPreKernel += profileThis(f->event);
	}

	/* Kernel Arguments
	 **/
	f->error = 
		clSetKernelArg(t->kernel[0], 0, sizeof(cl_mem), (void*)&t->data);
		$CHECKERROR
	f->error = 
		clSetKernelArg(t->kernel[0], 1, sizeof(cl_mem), (void*)&t->twiddle);
		$CHECKERROR
	f->error = 
		clSetKernelArg(t->kernel[0], 2, sizeof(int), (void*)&t->length);
		$CHECKERROR

	/* Bit Reversal */
	t->kernel_bit = malloc(sizeof(cl_kernel));
	t->kernel_bit[0] = clCreateKernel(	f->program1D, "reverse2", &f->error);
	$CHECKERROR
	
	f->error = 
		clSetKernelArg(	t->kernel_bit[0],0,sizeof(cl_mem), (void*)&t->bitrev[0]); 
		$CHECKERROR
	f->error = 
		clSetKernelArg(	t->kernel_bit[0],1,sizeof(int), (void*)&t->bits[0]);
		$CHECKERROR

	t->globalSize[0] = t->length/2;
	t->localSize[0] = 
		t->globalSize[0] == 1 ? 1 : // special: Input size 2
		t->length/2 < 512 ? t->length/4 : 256/2; // all others

	f->error = clEnqueueNDRangeKernel(	f->command_queue, t->kernel_bit[0],
										t->dim, NULL, t->globalSize,
										t->localSize, 0, NULL, &f->event);
	$CHECKERROR
	clFinish(f->command_queue);
	t->totalPreKernel += profileThis(f->event);
	
	#if 0 // Debug Code
	int *bit = malloc(sizeof(int)*t->length);
	f->error = 
		clEnqueueReadBuffer( f->command_queue, t->bitrev[0],
							 CL_TRUE, 0, sizeof(int)*t->length, bit,
							 0, NULL, NULL);
	$CHECKERROR
	int iii;
	for (iii = 0; iii < t->length; iii++) {
		printf("%d\n", bit[iii]);	
	}
	printf("----\n");
	exit(0);
	#endif
}

int findRadix(int n, int r){
	while(n>=r){
		if (n%r == 0)
		n=n/r;
		else
		break;
	}
	if(n==1)
		return r;
	return -1;
}

void findFactors(int *factArray, int n, int *radices, int rAvailable) {
	int xx, i = 0, s = n;
	for (xx = 0; xx < rAvailable; xx++) {
		if (n % radices[xx] == 0) {
			factArray[i] = 1;
			while (n >= radices[xx] ) {
				if (n % radices[xx] == 0) {
					factArray[i] *= radices[xx];
					n /= radices[xx];
				}
				else break;
			}
			i++;
			if (i == 2) {
				factArray[1] = s/factArray[0];
				return;
			}
		}
	}
	if (i == 1) { // if 1 factor found and 1 missing
		factArray[i] = n;
	}
	else if (i > 1) {
		factArray[1] = n;
	}
	else {
		factArray[0] = n;
		factArray[1] = -1;
	}
}

void Xtope1DPlanInit(struct topeFFT *f, 
					struct XtopePlan1D *t, 
					int x, int type, double *d) 
{
	/* Some Simple Initializations */
	t->totalMemory = t->totalKernel = t->totalPreKernel = 0;
	t->length 	= x;		// size
	t->dim 		= 1;		// Dimensions for kernel (will change if mix)
	t->type 	= type;		// C2C/R2C etc

	t->log 		= malloc(sizeof(int));
	t->radix 	= malloc(sizeof(int));
	t->side 	= malloc(sizeof(int));
	t->bits		= malloc(sizeof(int));

	/* Decide Radix 
	 **/
	int rAvailable = 8;
	int radices[8] = {10, 8 ,7, 6, 5, 4, 3, 2};
	int index;
	for(index=0; index < rAvailable; index++){
		t->radix[0] = findRadix(t->length,radices[index]);
		if(t->radix[0] != -1)
		break;
	}
	
	switch(t->radix[0])
	{
	case 10:
	case 7:
	case 6:
	case 5:
	case 3:
		t->log[0]  = log2(t->length)/log2(t->radix[0]);
		t->bits[0] = t->log[0];
		t->side[0] = t->length;
		break;
	case 8:
	case 4:
	case 2:
		t->log[0]  = log2(t->length)/log2(t->radix[0]);
		t->bits[0] = log2(t->length);
		t->side[0] = t->length;
		break;
	case -1:            //mix-radix
		t->dim   = 2;	// 2 factor support
		
		t->side  = realloc(t->side, sizeof(int)*t->dim);
		t->radix = realloc(t->radix,sizeof(int)*t->dim);
		t->log   = realloc(t->log,  sizeof(int)*t->dim);
		t->bits  = realloc(t->bits, sizeof(int)*t->dim);
		
		findFactors(t->side, t->length, radices, rAvailable);
		
		if(t->side[1] == -1) 	t->dim = 1;

		int ii;
		for (ii = 0; ii < t->dim; ii++) {
			for (index = 0; index < rAvailable; index++) {
				t->radix[ii] = findRadix(t->side[ii], radices[index]);
				if (t->radix[ii] != -1)
				break;
			}
			switch(t->radix[ii])
			{
			case 10:
			case 7:
			case 6:
			case 5:
			case 3: 
				t->bits[ii] = log2(t->side[ii])/log2(t->radix[ii]);
				t->log[ii]  = log2(t->side[ii])/log2(t->radix[ii]);
				break;
			case 8:
			case 4:
			case 2: 
				t->bits[ii] = log2(t->side[ii]);
				t->log[ii]  = log2(t->side[ii])/log2(t->radix[ii]);
				break;
			case -1:
				t->bits[ii] = t->log[ii] = -1;
			}
		}
		break;
	}

	t->globalSize = malloc(sizeof(size_t)*t->dim);	// Kernel indexing
	t->localSize  = malloc(sizeof(size_t)*t->dim);
	
	#if 1 // Debug code
	if (t->dim == 1) {
		fprintf(stderr,"Side:  %d\n", t->side[0]);
		fprintf(stderr,"Radix: %d\n", t->radix[0]);
		fprintf(stderr,"Bits : %d\n", t->bits[0]);
		fprintf(stderr,"Log  : %d\n", t->log[0]);
	}
	else {
		fprintf(stderr,"Sides: %d and %d\n", t->side[0], t->side[1]);
		fprintf(stderr,"Radix: %d and %d\n", t->radix[0], t->radix[1]);
		fprintf(stderr,"Bits : %d and %d\n", t->bits[0], t->bits[1]);
		fprintf(stderr,"Log  : %d and %d\n", t->log[0], t->log[1]);
	}
	#endif

	/* Memory Allocation for Data */
	t->dataSize = sizeof(double)*x*2;
	t->data   = clCreateBuffer(	f->context, CL_MEM_READ_WRITE,
								t->dataSize, NULL, &f->error);
	$CHECKERROR

	/* Memory allocation for bit reversal indices */
	int l;
	t->bitrev = malloc(sizeof(cl_mem)*t->dim);
	for (l = 0; l < t->dim; l++) {
		if (t->radix[l] > 0) {
			t->bitrev[l] = 
				clCreateBuffer( f->context, CL_MEM_READ_WRITE,	
								sizeof(int)*t->side[l], NULL, &f->error);
			$CHECKERROR
		}
	}

	/* Swapping Kernel Setup */
	if (t->dim == 1) {
		if (t->radix[0] > 0) { // for single radix algorithms
			t->kernel_swap = 
				clCreateKernel(f->program1D, "swap1D", &f->error);
				$CHECKERROR
			f->error =
				clSetKernelArg( t->kernel_swap,1,
								sizeof(cl_mem),(void*)&t->bitrev[0]); 
				$CHECKERROR
		}
		else {
			// no swapping req. for DFT
		}
	}
	else if (t->dim == 2) {
		if (t->radix[0] > 0 || t->radix[1] > 0) {
			t->kernel_swap = 
				clCreateKernel(f->program1D, "swapkernel", &f->error);
				$CHECKERROR
			f->error = 
				clSetKernelArg(t->kernel_swap,0,sizeof(cl_mem),(void*)&t->data);
				$CHECKERROR
			f->error = 
				clSetKernelArg(t->kernel_swap,1,sizeof(int),(void*)&t->side[0]);
				$CHECKERROR
			f->error = 
				clSetKernelArg(t->kernel_swap,2,sizeof(int),(void*)&t->side[1]);
				$CHECKERROR
		}
	}

	/* Divide Kernel for Inverse */
	t->kernel_div = clCreateKernel( f->program1D, "divide1D", &f->error);
	$CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,0,sizeof(cl_mem),
								(void*)&t->data); $CHECKERROR
	f->error = clSetKernelArg(	t->kernel_div,1,sizeof(int),
								(void*)&t->length); $CHECKERROR

	/* Send Rest of Setup to Right Functions*/
	if (t->dim == 1) {
		switch(t->radix[0])
		{
			case 8:
			case 4:
			case 2: 	Xtope1DPlanInitBase2(f,t); break;
			case 3:
			case 5:
			case 6:
			case 7: 	
			case 10: 	Xtope1DPlanInitBaseN(f,t); break;	
			case -1:    Xtope1DPlanInitDFT(f,t);   break;
		}
	}
	else if (t->dim == 2) {
		Xtope1dPlanInitMix(f,t);
	}

	/* Write Data */
	f->error = 
		clEnqueueWriteBuffer( f->command_queue, t->data, CL_TRUE, 
							  0, t->dataSize, d, 0, NULL, &f->event);
		$CHECKERROR
	clFinish(f->command_queue);
	t->totalMemory += profileThis(f->event);
}

void Xtope1DDestroy(	struct topeFFT *f,
						struct XtopePlan1D *t) 
{
	#if 0 // Fix these 
	if (t->x > 2) // Not initialized for under 2 
	{
		if (t->radix > 0) {
			f->error = clFlush(f->command_queue);
			f->error = clFinish(f->command_queue);
			f->error = clReleaseKernel(t->kernel);
			f->error = clReleaseKernel(t->kernel_bit); 
			f->error = clReleaseKernel(t->kernel_swap); 
			f->error = clReleaseKernel(t->kernel_twid);
			f->error = clReleaseKernel(t->kernel_div); 
			f->error = clReleaseProgram(f->program1D);
			f->error = clReleaseProgram(f->program2D);
			f->error = clReleaseProgram(f->program3D);
			f->error = clReleaseMemObject(t->data);
			f->error = clReleaseMemObject(t->bitrev); 
			f->error = clReleaseMemObject(t->twiddle); 
			f->error = clReleaseCommandQueue(f->command_queue);
			f->error = clReleaseContext(f->context);
		}
		else {
			f->error = clFlush(f->command_queue);
			f->error = clFinish(f->command_queue);
			f->error = clReleaseKernel(t->kernel);
			f->error = clReleaseProgram(f->program1D);
			f->error = clReleaseProgram(f->program2D);
			f->error = clReleaseProgram(f->program3D);
			f->error = clReleaseMemObject(t->data);
			f->error = clReleaseMemObject(t->scratch);
			f->error = clReleaseCommandQueue(f->command_queue);
			f->error = clReleaseContext(f->context);
		}
	}
	#endif
}

