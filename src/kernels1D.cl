#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define CLPI 3.141592653589793238462643383279 // acos(-1)
#define CLPT 6.283185307179586476925286766559 // acos(-1)*2
#define d707 0.707106781186547524400844362104 // cos(acos(-1)/4)

#define wa3 0.5
#define wb3 0.866025403784439

#define wa5 0.3090169944
#define wb5 0.9510565163
#define wc5 0.8090169944
#define wd5 0.5877852523

#define wa6 0.5
#define wb6 0.866025403784439

#define wa7 0.623489801858734
#define wb7 0.78183148246803
#define wc7 0.222520933956314
#define wd7 0.974927912181824
#define we7 0.900968867902419
#define wf7 0.433883739117558

#define wa10 0.809016994374947	
#define wb10 0.587785252292473
#define wc10 0.309016994374947
#define wd10 0.951056516295154

#define MAX  20;

int topePow(float x, int y, int levels) {
	#if 1
	int powLevels = y / levels;
	int powRemain = y % levels;
	int powX = 1;
	int l;
	for (l = 0; l < powLevels; l++)	
		powX *= pow(x,levels);
	powX *= pow(x,powRemain);
	#endif
	#if 0
	int powX = exp2(log2(2.)*stage);
	#endif
	return powX;
}

int topePowInt(int base,int exponent){
	int i=1;
	int power=1;

	for( i = 0; i < exponent; i++){

		   power=power*base;
	}

	return power;
}
double4 rad2(double4 LOC, double2 clSet1)
{
	double4 FIN =  (double4)(	LOC.x + LOC.z * clSet1.x - LOC.w * clSet1.y,
								LOC.y + LOC.w * clSet1.x + LOC.z * clSet1.y,
								LOC.x - LOC.z * clSet1.x + LOC.w * clSet1.y,
								LOC.y - LOC.w * clSet1.x - LOC.z * clSet1.y);
	return FIN;
}

__kernel void twid1D(__global double2 *twiddle, int size)
{
	int idX = get_global_id(0);
	twiddle[idX] =  (double2)(cos(CLPT*idX/size),-sin(CLPT*idX/size));
}

__kernel void DIT4C2C(	__global double *data, 
						__global double2 *twiddle,
						const int size, unsigned int stage,
						unsigned int dir,
						unsigned int useTwiddle) 
 
{
	int idX = get_global_id(0);
	
	int powX = topePow(4., stage, 7);
	int powXm1 = powX/4;

	int clipOne, clipTwo, clipThr, clipFou;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);

	double2 TEMPC;
	double8 SIGA = (double8)(	data[clipOne+0],data[clipOne+1],
									data[clipTwo+0],data[clipTwo+1],
									data[clipThr+0],data[clipThr+1],
									data[clipFou+0],data[clipFou+1]	);

	int coeffUse = kIndex * (size / powX);
	int red = size/4;
	double2 clSet1;
	int quad, buad;
	double2 clSet2, clSet3;

	if (useTwiddle == 1) {
		quad = coeffUse/red;
		buad = coeffUse%red;
		switch(quad) 
		{
			case 0:	clSet1 = (double2)(  twiddle[buad].x,  
										 twiddle[buad].y); break;
			case 1: clSet1 = (double2)(  twiddle[buad].y, 
									 	-twiddle[buad].x); break;
			case 2:	clSet1 = (double2)(	-twiddle[buad].x, 
										-twiddle[buad].y); break;
			case 3:	clSet1 = (double2)(	-twiddle[buad].y,  
										 twiddle[buad].x); break;
		}
		if (dir == 0) clSet1.y *= -1;
		if (kIndex != 0) {
			TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
			TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
			SIGA.s4 = TEMPC.x;
			SIGA.s5 = TEMPC.y;
		}

		quad = (2*coeffUse) / red;
		buad = (2*coeffUse) % red;
		switch(quad) 
		{
			case 0:	clSet1 = (double2)(  twiddle[buad].x,  
										 twiddle[buad].y); break;
			case 1: clSet1 = (double2)(  twiddle[buad].y, 
										-twiddle[buad].x); break;
			case 2:	clSet1 = (double2)(	-twiddle[buad].x, 
										-twiddle[buad].y); break;
			case 3:	clSet1 = (double2)(	-twiddle[buad].y,  
										 twiddle[buad].x); break;
		}
		if (dir == 0) clSet1.y *= -1;
		if (kIndex != 0) {
			TEMPC.x = SIGA.s2 * clSet1.x - SIGA.s3 * clSet1.y;
			TEMPC.y = SIGA.s3 * clSet1.x + SIGA.s2 * clSet1.y;
			SIGA.s2 = TEMPC.x;
			SIGA.s3 = TEMPC.y;
		}

		quad = (3*coeffUse) / red;
		buad = (3*coeffUse) % red;
		switch(quad) 
		{
			case 0:	clSet1 = (double2)(  twiddle[buad].x,  
										 twiddle[buad].y); break;
			case 1: clSet1 = (double2)(  twiddle[buad].y, 
										-twiddle[buad].x); break;
			case 2:	clSet1 = (double2)( -twiddle[buad].x, 
										-twiddle[buad].y); break;
			case 3:	clSet1 = (double2)( -twiddle[buad].y,  
										 twiddle[buad].x); break;
		}
		if (dir == 0) clSet1.y *= -1;
		if (kIndex != 0) {	
			TEMPC.x = SIGA.s6 * clSet1.x - SIGA.s7 * clSet1.y;
			TEMPC.y = SIGA.s7 * clSet1.x + SIGA.s6 * clSet1.y;
			SIGA.s6 = TEMPC.x;
			SIGA.s7 = TEMPC.y;
		}
	}
	else {
		if (kIndex != 0) {
			clSet2.x =  cos(2.*CLPT*kIndex/powX);
			clSet2.y = -sin(2.*CLPT*kIndex/powX);
			TEMPC.x = SIGA.s2 * clSet2.x - SIGA.s3 * clSet2.y;
			TEMPC.y = SIGA.s3 * clSet2.x + SIGA.s2 * clSet2.y;
			SIGA.s2 = TEMPC.x;
			SIGA.s3 = TEMPC.y;
			clSet1.x = cos(CLPT*kIndex/powX);
			clSet1.y = -sin(CLPT*kIndex/powX);
			TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
			TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
			SIGA.s4 = TEMPC.x;
			SIGA.s5 = TEMPC.y;
			clSet3.x = cos(3.*CLPT*kIndex/powX);
			clSet3.y = -sin(3.*CLPT*kIndex/powX);
			TEMPC.x = SIGA.s6 * clSet3.x - SIGA.s7 * clSet3.y;
			TEMPC.y = SIGA.s7 * clSet3.x + SIGA.s6 * clSet3.y;
			SIGA.s6 = TEMPC.x;
			SIGA.s7 = TEMPC.y;
		}	
	}
	
	if (dir == 1) {
		data[clipOne+0] = SIGA.s0 + SIGA.s2 + SIGA.s4 + SIGA.s6;
		data[clipOne+1] = SIGA.s1 + SIGA.s3 + SIGA.s5 + SIGA.s7;
		data[clipTwo+0] = SIGA.s0 - SIGA.s2 + SIGA.s5 - SIGA.s7;
		data[clipTwo+1] = SIGA.s1 - SIGA.s3 - SIGA.s4 + SIGA.s6;
		data[clipThr+0] = SIGA.s0 + SIGA.s2 - SIGA.s4 - SIGA.s6;
		data[clipThr+1] = SIGA.s1 + SIGA.s3 - SIGA.s5 - SIGA.s7;
		data[clipFou+0] = SIGA.s0 - SIGA.s2 - SIGA.s5 + SIGA.s7;
		data[clipFou+1] = SIGA.s1 - SIGA.s3 + SIGA.s4 - SIGA.s6;
	}
	else if (dir == 0) {
		data[clipOne+0] = SIGA.s0 + SIGA.s2 + SIGA.s4 + SIGA.s6;
		data[clipOne+1] = SIGA.s1 + SIGA.s3 + SIGA.s5 + SIGA.s7;
		data[clipTwo+0] = SIGA.s0 - SIGA.s2 - SIGA.s5 + SIGA.s7;
		data[clipTwo+1] = SIGA.s1 - SIGA.s3 + SIGA.s4 - SIGA.s6;
		data[clipThr+0] = SIGA.s0 + SIGA.s2 - SIGA.s4 - SIGA.s6;
		data[clipThr+1] = SIGA.s1 + SIGA.s3 - SIGA.s5 - SIGA.s7;
		data[clipFou+0] = SIGA.s0 - SIGA.s2 + SIGA.s5 - SIGA.s7;
		data[clipFou+1] = SIGA.s1 - SIGA.s3 - SIGA.s4 + SIGA.s6;
	}
	#if 0
	data[clipOne+0] = 0;//
	data[clipOne+1] = 0;
	data[clipTwo+0] = clSet2.x;
	data[clipTwo+1] = clSet2.y;
	data[clipThr+0] = clSet1.x;
	data[clipThr+1] = clSet1.y;
	data[clipFou+0] = clSet3.x;
	data[clipFou+1] = clSet3.y;
	#endif
}


__kernel void DIT10C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int powX = topePow(10.,stage,1);
	int powXm1 = powX/10;

	int clipOne, clipTwo, clipThr, clipFou, clipFiv, 
		clipSix, clipSev, clipEight, clipNine, clipTen;
	int yIndex, kIndex;

	int BASE 	= 0;
	int STRIDE 	= 1;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}
	
	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));
	clipFiv		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 4 * powXm1));
	clipSix		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 5 * powXm1));
	clipSev		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 6 * powXm1));
	clipEight	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 7 * powXm1));
	clipNine	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 8 * powXm1));
	clipTen		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 9 * powXm1));

	
	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
	double2 SIG6A = (double2)(	data[clipSix+0],data[clipSix+1]);
	double2 SIG7A = (double2)(	data[clipSev+0],data[clipSev+1]);
	double2 SIG8A = (double2)(	data[clipEight+0],data[clipEight+1]);
	double2 SIG9A = (double2)(	data[clipNine+0],data[clipNine+1]);
	double2 SIG10A = (double2)(	data[clipTen+0],data[clipTen+1]);
	
	double2 clSet2, clSet3, clSet4, clSet5, clSet6, clSet7, clSet8, clSet9, clSet10, temp2;
	
	if (kIndex!=0) {
		
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
	
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;

		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
	
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;

		clSet6.x = cos(5*CLPT*kIndex/powX);
		clSet6.y = -sin(5*CLPT*kIndex/powX);
		if (dir == 0) clSet6.y *= -1;
		TEMPC.x = SIG6A.x * clSet6.x - SIG6A.y * clSet6.y;
		TEMPC.y = SIG6A.x * clSet6.y + SIG6A.y * clSet6.x;
		SIG6A.x = TEMPC.x;
		SIG6A.y = TEMPC.y;

		clSet7.x = cos(6*CLPT*kIndex/powX);
		clSet7.y = -sin(6*CLPT*kIndex/powX);
		if (dir == 0) clSet7.y *= -1;
		TEMPC.x = SIG7A.x * clSet7.x - SIG7A.y * clSet7.y;
		TEMPC.y = SIG7A.x * clSet7.y + SIG7A.y * clSet7.x;
		SIG7A.x = TEMPC.x;
		SIG7A.y = TEMPC.y;

		
		clSet8.x = cos(7*CLPT*kIndex/powX);
		clSet8.y = -sin(7*CLPT*kIndex/powX);
		if (dir == 0) clSet8.y *= -1;
		TEMPC.x = SIG8A.x * clSet8.x - SIG8A.y * clSet8.y;
		TEMPC.y = SIG8A.x * clSet8.y + SIG8A.y * clSet8.x;
		SIG8A.x = TEMPC.x;
		SIG8A.y = TEMPC.y;

		
		clSet9.x = cos(8*CLPT*kIndex/powX);
		clSet9.y = -sin(8*CLPT*kIndex/powX);
		if (dir == 0) clSet9.y *= -1;
		TEMPC.x = SIG9A.x * clSet9.x - SIG9A.y * clSet9.y;
		TEMPC.y = SIG9A.x * clSet9.y + SIG9A.y * clSet9.x;
		SIG9A.x = TEMPC.x;
		SIG9A.y = TEMPC.y;

		
		clSet10.x = cos(9*CLPT*kIndex/powX);
		clSet10.y = -sin(9*CLPT*kIndex/powX);
		if (dir == 0) clSet10.y *= -1;
		TEMPC.x = SIG10A.x * clSet10.x - SIG10A.y * clSet10.y;
		TEMPC.y = SIG10A.x * clSet10.y + SIG10A.y * clSet10.x;
		SIG10A.x = TEMPC.x;
		SIG10A.y = TEMPC.y;
	


		
	}	
		data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x + SIG6A.x + SIG7A.x + SIG8A.x + SIG9A.x + SIG10A.x;
		data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y + SIG6A.y + SIG7A.y + SIG8A.y + SIG9A.y + SIG10A.y;
	
		data[clipTwo+0] = SIG4A.s0 + (wa10*SIG4A.s2 + wb10*SIG4A.s3) +(wc10*SIG4A.s4 + wd10*SIG4A.s5) + (-wc10*SIG4A.s6 + wd10*SIG4A.s7) + (-wa10*SIG5A.x + wb10*SIG5A.y) + (-SIG6A.x) + (-wa10*SIG7A.x - wb10*SIG7A.y) + (-wc10*SIG8A.x - wd10*SIG8A.y) + (wc10*SIG9A.x - wd10*SIG9A.y) + (wa10*SIG10A.x - wb10*SIG10A.y);
		data[clipTwo+1] = SIG4A.s1 + (wa10*SIG4A.s3 - wb10*SIG4A.s2) +(wc10*SIG4A.s5 - wd10*SIG4A.s4) + (-wc10*SIG4A.s7 - wd10*SIG4A.s6) + (-wa10*SIG5A.y - wb10*SIG5A.x) + (-SIG6A.y) + (-wa10*SIG7A.y + wb10*SIG7A.x) + (-wc10*SIG8A.y + wd10*SIG8A.x) + (wc10*SIG9A.y + wd10*SIG9A.x) +(wa10*SIG10A.y + wb10*SIG10A.x) ;

		data[clipThr+0] = SIG4A.s0 + (wc10*SIG4A.s2 + wd10*SIG4A.s3) +(-wa10*SIG4A.s4 + wb10*SIG4A.s5) + (-wa10*SIG4A.s6 - wb10*SIG4A.s7) + (wc10*SIG5A.x - wd10*SIG5A.y) + (SIG6A.x) + (wc10*SIG7A.x + 	wd10*SIG7A.y) + (-wa10*SIG8A.x + wb10*SIG8A.y) + (-wa10*SIG9A.x - wb10*SIG9A.y) + (wc10*SIG10A.x - wd10*SIG10A.y);
		data[clipThr+1] = SIG4A.s1 + (wc10*SIG4A.s3 - wd10*SIG4A.s2) +(-wa10*SIG4A.s5 - wb10*SIG4A.s4) + (-wa10*SIG4A.s7 + wb10*SIG4A.s6) + (wc10*SIG5A.y + wd10*SIG5A.x) + (SIG6A.y) + (wc10*SIG7A.y - wd10*SIG7A.x) + (-wa10*SIG8A.y - wb10*SIG8A.x) + (-wa10*SIG9A.y + wb10*SIG9A.x) +(wc10*SIG10A.y + wd10*SIG10A.x) ;


	data[clipFou+0] = SIG4A.s0 + (-wc10*SIG4A.s2 + wd10*SIG4A.s3) 
					+(-wa10*SIG4A.s4 - wb10*SIG4A.s5) 
					+ (wa10*SIG4A.s6 - wb10*SIG4A.s7) 
					+ (wc10*SIG5A.x + wd10*SIG5A.y) 
					+ (-SIG6A.x) + (wc10*SIG7A.x - 	wd10*SIG7A.y) 
					+ (wa10*SIG8A.x + wb10*SIG8A.y) 
					+ (-wa10*SIG9A.x + wb10*SIG9A.y) 
					+ (-wc10*SIG10A.x - wd10*SIG10A.y);
	data[clipFou+1] = SIG4A.s1 + (-wc10*SIG4A.s3 - wd10*SIG4A.s2) 
					+(-wa10*SIG4A.s5 + wb10*SIG4A.s4) 
					+ (wa10*SIG4A.s7 + wb10*SIG4A.s6) 
					+ (wc10*SIG5A.y - wd10*SIG5A.x) + (-SIG6A.y) 
					+ (wc10*SIG7A.y + wd10*SIG7A.x) 
					+ (wa10*SIG8A.y - wb10*SIG8A.x) 
					+ (-wa10*SIG9A.y - wb10*SIG9A.x) 
					+(-wc10*SIG10A.y + wd10*SIG10A.x) ;

	data[clipFiv+0] = SIG4A.s0 + (-wa10*SIG4A.s2 + wb10*SIG4A.s3) 
					+ (wc10*SIG4A.s4 - wd10*SIG4A.s5) 
					+ (wc10*SIG4A.s6 + wd10*SIG4A.s7) 
					+ (-wa10*SIG5A.x - wb10*SIG5A.y) 
					+ (SIG6A.x) + (-wa10*SIG7A.x + 	wb10*SIG7A.y) 
					+ (wc10*SIG8A.x - wd10*SIG8A.y) 
					+ (wc10*SIG9A.x + wd10*SIG9A.y) 
					+ (-wa10*SIG10A.x - wb10*SIG10A.y);
	data[clipFiv+1] = SIG4A.s1 + (-wa10*SIG4A.s3 - wb10*SIG4A.s2) 
					+ (wc10*SIG4A.s5 + wd10*SIG4A.s4) 
					+ (wc10*SIG4A.s7 - wd10*SIG4A.s6) 
					+ (-wa10*SIG5A.y + wb10*SIG5A.x) 
					+ (SIG6A.y) + (-wa10*SIG7A.y - wb10*SIG7A.x) 
					+ (wc10*SIG8A.y + wd10*SIG8A.x) 
					+ (wc10*SIG9A.y - wd10*SIG9A.x) 
					+(-wa10*SIG10A.y + wb10*SIG10A.x) ;

		data[clipSix+0] = SIG4A.s0 + (-SIG4A.s2) + SIG4A.s4 + (-SIG4A.s6) + SIG5A.x + (-SIG6A.x) + SIG7A.x + (-SIG8A.x) + SIG9A.x + (-SIG10A.x);
		data[clipSix+1] = SIG4A.s1 + (-SIG4A.s3) + SIG4A.s5 + (-SIG4A.s7) + SIG5A.y + (-SIG6A.y) + SIG7A.y + (-SIG8A.y) + SIG9A.y + (-SIG10A.y);

		data[clipSev+0] = SIG4A.s0 + (-wa10*SIG4A.s2 - wb10*SIG4A.s3) +(wc10*SIG4A.s4 + wd10*SIG4A.s5) + (wc10*SIG4A.s6 - wd10*SIG4A.s7) + (-wa10*SIG5A.x + wb10*SIG5A.y) + (SIG6A.x) + (-wa10*SIG7A.x - wb10*SIG7A.y) + (wc10*SIG8A.x + wd10*SIG8A.y) + (wc10*SIG9A.x - wd10*SIG9A.y) + (-wa10*SIG10A.x + wb10*SIG10A.y);
		data[clipSev+1] = SIG4A.s1 + (-wa10*SIG4A.s3 + wb10*SIG4A.s2) +(wc10*SIG4A.s5 - wd10*SIG4A.s4) + (wc10*SIG4A.s7 + wd10*SIG4A.s6) + (-wa10*SIG5A.y - wb10*SIG5A.x) + (SIG6A.y) + (-wa10*SIG7A.y + wb10*SIG7A.x) + (wc10*SIG8A.y - wd10*SIG8A.x) + (wc10*SIG9A.y + wd10*SIG9A.x) +(-wa10*SIG10A.y - wb10*SIG10A.x) ;


		data[clipEight+0] = SIG4A.s0 + (-wc10*SIG4A.s2 - wd10*SIG4A.s3) +(-wa10*SIG4A.s4 + wb10*SIG4A.s5) + (wa10*SIG4A.s6 + wb10*SIG4A.s7) + (wc10*SIG5A.x - wd10*SIG5A.y) + (-SIG6A.x) + (wc10*SIG7A.x + wd10*SIG7A.y) + (wa10*SIG8A.x - wb10*SIG8A.y) + (-wa10*SIG9A.x - wb10*SIG9A.y) + (-wc10*SIG10A.x + wd10*SIG10A.y);
		data[clipEight+1] = SIG4A.s1 + (-wc10*SIG4A.s3 + wd10*SIG4A.s2) +(-wa10*SIG4A.s5 -wb10*SIG4A.s4) + (wa10*SIG4A.s7 - wb10*SIG4A.s6) + (wc10*SIG5A.y + wd10*SIG5A.x) + (-SIG6A.y) + (wc10*SIG7A.y - wd10*SIG7A.x) + (wa10*SIG8A.y + wb10*SIG8A.x) + (-wa10*SIG9A.y + wb10*SIG9A.x) +(-wc10*SIG10A.y - wd10*SIG10A.x) ;


		data[clipNine+0] = SIG4A.s0 + (wc10*SIG4A.s2 - wd10*SIG4A.s3) +(-wa10*SIG4A.s4 - wb10*SIG4A.s5) + (-wa10*SIG4A.s6 + wb10*SIG4A.s7) + (wc10*SIG5A.x + wd10*SIG5A.y) + (SIG6A.x) + (wc10*SIG7A.x - wd10*SIG7A.y) + (-wa10*SIG8A.x - wb10*SIG8A.y) + (-wa10*SIG9A.x + wb10*SIG9A.y) + (wc10*SIG10A.x + wd10*SIG10A.y);
		data[clipNine+1] = SIG4A.s1 + (wc10*SIG4A.s3 + wd10*SIG4A.s2) +(-wa10*SIG4A.s5 + wb10*SIG4A.s4) + (-wa10*SIG4A.s7 - wb10*SIG4A.s6) + (wc10*SIG5A.y - wd10*SIG5A.x) + (SIG6A.y) + (wc10*SIG7A.y + wd10*SIG7A.x) + (-wa10*SIG8A.y + wb10*SIG8A.x) + (-wa10*SIG9A.y - wb10*SIG9A.x) +(wc10*SIG10A.y - wd10*SIG10A.x) ;


		data[clipTen+0] = SIG4A.s0 + (wa10*SIG4A.s2 - wb10*SIG4A.s3) +(wc10*SIG4A.s4 - wd10*SIG4A.s5) + (-wc10*SIG4A.s6 - wd10*SIG4A.s7) + (-wa10*SIG5A.x - wb10*SIG5A.y) + (-SIG6A.x) + (-wa10*SIG7A.x + wb10*SIG7A.y) + (-wc10*SIG8A.x + wd10*SIG8A.y) + (wc10*SIG9A.x + wd10*SIG9A.y) + (wa10*SIG10A.x + wb10*SIG10A.y);
		data[clipTen+1] = SIG4A.s1 + (wa10*SIG4A.s3 + wb10*SIG4A.s2) +(wc10*SIG4A.s5 + wd10*SIG4A.s4) + (-wc10*SIG4A.s7 + wd10*SIG4A.s6) + (-wa10*SIG5A.y + wb10*SIG5A.x) + (-SIG6A.y) + (-wa10*SIG7A.y - wb10*SIG7A.x) + (-wc10*SIG8A.y - wd10*SIG8A.x) + (wc10*SIG9A.y - wd10*SIG9A.x) +(wa10*SIG10A.y - wb10*SIG10A.x) ;
	
	#if 0 // Debug code
	data[clipOne+0] = powX;//kIndex;
	data[clipOne+1] = 11;//yIndex;
	data[clipTwo+0] = powX;//kIndex;
	data[clipTwo+1] = 22;//yIndex;
	data[clipThr+0] = powX;//kIndex;
	data[clipThr+1] = 33;//yIndex;
	data[clipFou+0] = powX;//kIndex;
	data[clipFou+1] = 44;//yIndex;
	data[clipFiv+0] = powX;//kIndex;
	data[clipFiv+1] = 55;//yIndex;
	data[clipSix+0] = powX;//kIndex;
	data[clipSix+1] = 66;//yIndex;
	data[clipSev+0] = powX;//kIndex;
	data[clipSev+1] = 77;//yIndex;
	data[clipEight+0] = powX;//kIndex;
	data[clipEight+1] = 88;//yIndex;
	data[clipNine+0] = powX;//kIndex;
	data[clipNine+1] = 99;//yIndex;
	data[clipTen+0] = powX;//kIndex;
	data[clipTen+1] = 100;//yIndex;
	#endif
}

__kernel void DIT10C2C(	__global double *data, 
			const int size,
			unsigned int stage,
			unsigned int dir)
{
#if 1 // correct
	int idX = get_global_id(0);
	int powMaxLvl = 3;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(10.0,powMaxLvl);
	}
	powX *= pow(10.0,powRemain);
	powXm1 = powX/10;
	#if 0
	int powX = exp2(log2(5.)*stage);
	int powXm1 = powX/5;
	#endif       
	// x =pow(5.0,2)=25 whereas x=pow(5.0f,2)=24												
	
	int clipOne, clipTwo, clipThr, clipFou, clipFiv, clipSix, clipSev, clipEight, clipNine, clipTen;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);
	clipFiv		= 2 * (kIndex + yIndex * powX + 4 * powXm1);
	clipSix		= 2 * (kIndex + yIndex * powX + 5 * powXm1);
	clipSev		= 2 * (kIndex + yIndex * powX + 6 * powXm1);
	clipEight	= 2 * (kIndex + yIndex * powX + 7 * powXm1);
	clipNine	= 2 * (kIndex + yIndex * powX + 8 * powXm1);
	clipTen		= 2 * (kIndex + yIndex * powX + 9 * powXm1);

	
	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
	double2 SIG6A = (double2)(	data[clipSix+0],data[clipSix+1]);
	double2 SIG7A = (double2)(	data[clipSev+0],data[clipSev+1]);
	double2 SIG8A = (double2)(	data[clipEight+0],data[clipEight+1]);
	double2 SIG9A = (double2)(	data[clipNine+0],data[clipNine+1]);
	double2 SIG10A = (double2)(	data[clipTen+0],data[clipTen+1]);
	
	int coeffUse = kIndex * (size / powX);
	double2 clSet2, clSet3, clSet4, clSet5, clSet6, clSet7, clSet8, clSet9, clSet10, temp2;
	
	if (kIndex!=0) {
		
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
	
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;

		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
	
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;

		clSet6.x = cos(5*CLPT*kIndex/powX);
		clSet6.y = -sin(5*CLPT*kIndex/powX);
		if (dir == 0) clSet6.y *= -1;
		TEMPC.x = SIG6A.x * clSet6.x - SIG6A.y * clSet6.y;
		TEMPC.y = SIG6A.x * clSet6.y + SIG6A.y * clSet6.x;
		SIG6A.x = TEMPC.x;
		SIG6A.y = TEMPC.y;

		clSet7.x = cos(6*CLPT*kIndex/powX);
		clSet7.y = -sin(6*CLPT*kIndex/powX);
		if (dir == 0) clSet7.y *= -1;
		TEMPC.x = SIG7A.x * clSet7.x - SIG7A.y * clSet7.y;
		TEMPC.y = SIG7A.x * clSet7.y + SIG7A.y * clSet7.x;
		SIG7A.x = TEMPC.x;
		SIG7A.y = TEMPC.y;

		
		clSet8.x = cos(7*CLPT*kIndex/powX);
		clSet8.y = -sin(7*CLPT*kIndex/powX);
		if (dir == 0) clSet8.y *= -1;
		TEMPC.x = SIG8A.x * clSet8.x - SIG8A.y * clSet8.y;
		TEMPC.y = SIG8A.x * clSet8.y + SIG8A.y * clSet8.x;
		SIG8A.x = TEMPC.x;
		SIG8A.y = TEMPC.y;

		
		clSet9.x = cos(8*CLPT*kIndex/powX);
		clSet9.y = -sin(8*CLPT*kIndex/powX);
		if (dir == 0) clSet9.y *= -1;
		TEMPC.x = SIG9A.x * clSet9.x - SIG9A.y * clSet9.y;
		TEMPC.y = SIG9A.x * clSet9.y + SIG9A.y * clSet9.x;
		SIG9A.x = TEMPC.x;
		SIG9A.y = TEMPC.y;

		
		clSet10.x = cos(9*CLPT*kIndex/powX);
		clSet10.y = -sin(9*CLPT*kIndex/powX);
		if (dir == 0) clSet10.y *= -1;
		TEMPC.x = SIG10A.x * clSet10.x - SIG10A.y * clSet10.y;
		TEMPC.y = SIG10A.x * clSet10.y + SIG10A.y * clSet10.x;
		SIG10A.x = TEMPC.x;
		SIG10A.y = TEMPC.y;
	


		
	}	
		data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x + SIG6A.x + SIG7A.x + SIG8A.x + SIG9A.x + SIG10A.x;
		data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y + SIG6A.y + SIG7A.y + SIG8A.y + SIG9A.y + SIG10A.y;
	
		data[clipTwo+0] = SIG4A.s0 + (wa10*SIG4A.s2 + wb10*SIG4A.s3) +(wc10*SIG4A.s4 + wd10*SIG4A.s5) + (-wc10*SIG4A.s6 + wd10*SIG4A.s7) + (-wa10*SIG5A.x + wb10*SIG5A.y) + (-SIG6A.x) + (-wa10*SIG7A.x - wb10*SIG7A.y) + (-wc10*SIG8A.x - wd10*SIG8A.y) + (wc10*SIG9A.x - wd10*SIG9A.y) + (wa10*SIG10A.x - wb10*SIG10A.y);
		data[clipTwo+1] = SIG4A.s1 + (wa10*SIG4A.s3 - wb10*SIG4A.s2) +(wc10*SIG4A.s5 - wd10*SIG4A.s4) + (-wc10*SIG4A.s7 - wd10*SIG4A.s6) + (-wa10*SIG5A.y - wb10*SIG5A.x) + (-SIG6A.y) + (-wa10*SIG7A.y + wb10*SIG7A.x) + (-wc10*SIG8A.y + wd10*SIG8A.x) + (wc10*SIG9A.y + wd10*SIG9A.x) +(wa10*SIG10A.y + wb10*SIG10A.x) ;

		data[clipThr+0] = SIG4A.s0 + (wc10*SIG4A.s2 + wd10*SIG4A.s3) +(-wa10*SIG4A.s4 + wb10*SIG4A.s5) + (-wa10*SIG4A.s6 - wb10*SIG4A.s7) + (wc10*SIG5A.x - wd10*SIG5A.y) + (SIG6A.x) + (wc10*SIG7A.x + 	wd10*SIG7A.y) + (-wa10*SIG8A.x + wb10*SIG8A.y) + (-wa10*SIG9A.x - wb10*SIG9A.y) + (wc10*SIG10A.x - wd10*SIG10A.y);
		data[clipThr+1] = SIG4A.s1 + (wc10*SIG4A.s3 - wd10*SIG4A.s2) +(-wa10*SIG4A.s5 - wb10*SIG4A.s4) + (-wa10*SIG4A.s7 + wb10*SIG4A.s6) + (wc10*SIG5A.y + wd10*SIG5A.x) + (SIG6A.y) + (wc10*SIG7A.y - wd10*SIG7A.x) + (-wa10*SIG8A.y - wb10*SIG8A.x) + (-wa10*SIG9A.y + wb10*SIG9A.x) +(wc10*SIG10A.y + wd10*SIG10A.x) ;


	data[clipFou+0] = SIG4A.s0 + (-wc10*SIG4A.s2 + wd10*SIG4A.s3) 
					+(-wa10*SIG4A.s4 - wb10*SIG4A.s5) 
					+ (wa10*SIG4A.s6 - wb10*SIG4A.s7) 
					+ (wc10*SIG5A.x + wd10*SIG5A.y) 
					+ (-SIG6A.x) + (wc10*SIG7A.x - 	wd10*SIG7A.y) 
					+ (wa10*SIG8A.x + wb10*SIG8A.y) 
					+ (-wa10*SIG9A.x + wb10*SIG9A.y) 
					+ (-wc10*SIG10A.x - wd10*SIG10A.y);
	data[clipFou+1] = SIG4A.s1 + (-wc10*SIG4A.s3 - wd10*SIG4A.s2) 
					+(-wa10*SIG4A.s5 + wb10*SIG4A.s4) 
					+ (wa10*SIG4A.s7 + wb10*SIG4A.s6) 
					+ (wc10*SIG5A.y - wd10*SIG5A.x) + (-SIG6A.y) 
					+ (wc10*SIG7A.y + wd10*SIG7A.x) 
					+ (wa10*SIG8A.y - wb10*SIG8A.x) 
					+ (-wa10*SIG9A.y - wb10*SIG9A.x) 
					+(-wc10*SIG10A.y + wd10*SIG10A.x) ;

	data[clipFiv+0] = SIG4A.s0 + (-wa10*SIG4A.s2 + wb10*SIG4A.s3) 
					+ (wc10*SIG4A.s4 - wd10*SIG4A.s5) 
					+ (wc10*SIG4A.s6 + wd10*SIG4A.s7) 
					+ (-wa10*SIG5A.x - wb10*SIG5A.y) 
					+ (SIG6A.x) + (-wa10*SIG7A.x + 	wb10*SIG7A.y) 
					+ (wc10*SIG8A.x - wd10*SIG8A.y) 
					+ (wc10*SIG9A.x + wd10*SIG9A.y) 
					+ (-wa10*SIG10A.x - wb10*SIG10A.y);
	data[clipFiv+1] = SIG4A.s1 + (-wa10*SIG4A.s3 - wb10*SIG4A.s2) 
					+ (wc10*SIG4A.s5 + wd10*SIG4A.s4) 
					+ (wc10*SIG4A.s7 - wd10*SIG4A.s6) 
					+ (-wa10*SIG5A.y + wb10*SIG5A.x) 
					+ (SIG6A.y) + (-wa10*SIG7A.y - wb10*SIG7A.x) 
					+ (wc10*SIG8A.y + wd10*SIG8A.x) 
					+ (wc10*SIG9A.y - wd10*SIG9A.x) 
					+(-wa10*SIG10A.y + wb10*SIG10A.x) ;

		data[clipSix+0] = SIG4A.s0 + (-SIG4A.s2) + SIG4A.s4 + (-SIG4A.s6) + SIG5A.x + (-SIG6A.x) + SIG7A.x + (-SIG8A.x) + SIG9A.x + (-SIG10A.x);
		data[clipSix+1] = SIG4A.s1 + (-SIG4A.s3) + SIG4A.s5 + (-SIG4A.s7) + SIG5A.y + (-SIG6A.y) + SIG7A.y + (-SIG8A.y) + SIG9A.y + (-SIG10A.y);

		data[clipSev+0] = SIG4A.s0 + (-wa10*SIG4A.s2 - wb10*SIG4A.s3) +(wc10*SIG4A.s4 + wd10*SIG4A.s5) + (wc10*SIG4A.s6 - wd10*SIG4A.s7) + (-wa10*SIG5A.x + wb10*SIG5A.y) + (SIG6A.x) + (-wa10*SIG7A.x - wb10*SIG7A.y) + (wc10*SIG8A.x + wd10*SIG8A.y) + (wc10*SIG9A.x - wd10*SIG9A.y) + (-wa10*SIG10A.x + wb10*SIG10A.y);
		data[clipSev+1] = SIG4A.s1 + (-wa10*SIG4A.s3 + wb10*SIG4A.s2) +(wc10*SIG4A.s5 - wd10*SIG4A.s4) + (wc10*SIG4A.s7 + wd10*SIG4A.s6) + (-wa10*SIG5A.y - wb10*SIG5A.x) + (SIG6A.y) + (-wa10*SIG7A.y + wb10*SIG7A.x) + (wc10*SIG8A.y - wd10*SIG8A.x) + (wc10*SIG9A.y + wd10*SIG9A.x) +(-wa10*SIG10A.y - wb10*SIG10A.x) ;


		data[clipEight+0] = SIG4A.s0 + (-wc10*SIG4A.s2 - wd10*SIG4A.s3) +(-wa10*SIG4A.s4 + wb10*SIG4A.s5) + (wa10*SIG4A.s6 + wb10*SIG4A.s7) + (wc10*SIG5A.x - wd10*SIG5A.y) + (-SIG6A.x) + (wc10*SIG7A.x + wd10*SIG7A.y) + (wa10*SIG8A.x - wb10*SIG8A.y) + (-wa10*SIG9A.x - wb10*SIG9A.y) + (-wc10*SIG10A.x + wd10*SIG10A.y);
		data[clipEight+1] = SIG4A.s1 + (-wc10*SIG4A.s3 + wd10*SIG4A.s2) +(-wa10*SIG4A.s5 -wb10*SIG4A.s4) + (wa10*SIG4A.s7 - wb10*SIG4A.s6) + (wc10*SIG5A.y + wd10*SIG5A.x) + (-SIG6A.y) + (wc10*SIG7A.y - wd10*SIG7A.x) + (wa10*SIG8A.y + wb10*SIG8A.x) + (-wa10*SIG9A.y + wb10*SIG9A.x) +(-wc10*SIG10A.y - wd10*SIG10A.x) ;


		data[clipNine+0] = SIG4A.s0 + (wc10*SIG4A.s2 - wd10*SIG4A.s3) +(-wa10*SIG4A.s4 - wb10*SIG4A.s5) + (-wa10*SIG4A.s6 + wb10*SIG4A.s7) + (wc10*SIG5A.x + wd10*SIG5A.y) + (SIG6A.x) + (wc10*SIG7A.x - wd10*SIG7A.y) + (-wa10*SIG8A.x - wb10*SIG8A.y) + (-wa10*SIG9A.x + wb10*SIG9A.y) + (wc10*SIG10A.x + wd10*SIG10A.y);
		data[clipNine+1] = SIG4A.s1 + (wc10*SIG4A.s3 + wd10*SIG4A.s2) +(-wa10*SIG4A.s5 + wb10*SIG4A.s4) + (-wa10*SIG4A.s7 - wb10*SIG4A.s6) + (wc10*SIG5A.y - wd10*SIG5A.x) + (SIG6A.y) + (wc10*SIG7A.y + wd10*SIG7A.x) + (-wa10*SIG8A.y + wb10*SIG8A.x) + (-wa10*SIG9A.y - wb10*SIG9A.x) +(wc10*SIG10A.y - wd10*SIG10A.x) ;


		data[clipTen+0] = SIG4A.s0 + (wa10*SIG4A.s2 - wb10*SIG4A.s3) +(wc10*SIG4A.s4 - wd10*SIG4A.s5) + (-wc10*SIG4A.s6 - wd10*SIG4A.s7) + (-wa10*SIG5A.x - wb10*SIG5A.y) + (-SIG6A.x) + (-wa10*SIG7A.x + wb10*SIG7A.y) + (-wc10*SIG8A.x + wd10*SIG8A.y) + (wc10*SIG9A.x + wd10*SIG9A.y) + (wa10*SIG10A.x + wb10*SIG10A.y);
		data[clipTen+1] = SIG4A.s1 + (wa10*SIG4A.s3 + wb10*SIG4A.s2) +(wc10*SIG4A.s5 + wd10*SIG4A.s4) + (-wc10*SIG4A.s7 + wd10*SIG4A.s6) + (-wa10*SIG5A.y + wb10*SIG5A.x) + (-SIG6A.y) + (-wa10*SIG7A.y - wb10*SIG7A.x) + (-wc10*SIG8A.y - wd10*SIG8A.x) + (wc10*SIG9A.y - wd10*SIG9A.x) +(wa10*SIG10A.y - wb10*SIG10A.x) ;
	
	#if 0 // Debug code
	data[clipOne+0] = 11;//kIndex;
	data[clipOne+1] = 11;//yIndex;
	data[clipTwo+0] = 22;//kIndex;
	data[clipTwo+1] = 22;//yIndex;
	data[clipThr+0] = 33;//kIndex;
	data[clipThr+1] = 33;//yIndex;
	data[clipFou+0] = 44;//kIndex;
	data[clipFou+1] = 44;//yIndex;
	data[clipFiv+0] = 55;//kIndex;
	data[clipFiv+1] = 55;//yIndex;
	data[clipSix+0] = 66;//kIndex;
	data[clipSix+1] = 66;//yIndex;
	data[clipSev+0] = 77;//kIndex;
	data[clipSev+1] = 77;//yIndex;
	data[clipEight+0] = 88;//kIndex;
	data[clipEight+1] = 88;//yIndex;
	data[clipNine+0] = 99;//kIndex;
	data[clipNine+1] = 99;//yIndex;
	data[clipTen+0] = 100;//kIndex;
	data[clipTen+1] = 100;//yIndex;
	#endif
#endif //end correct
}


__kernel void DIT8C2C(	__global double *data, 
						__global double2 *twiddle,
						const int size, unsigned int stage,
						unsigned int dir,
						unsigned int useTwiddle) 
 
{
	int idX = get_global_id(0);

	int powX = topePow(8.,stage,4);
	int powXm1 = powX/8;

	int clipOne, clipTwo, clipThr, clipFou, clipFiv, clipSix, clipSev, clipEig;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);
	clipFiv		= 2 * (kIndex + yIndex * powX + 4 * powXm1);
	clipSix		= 2 * (kIndex + yIndex * powX + 5 * powXm1);
	clipSev		= 2 * (kIndex + yIndex * powX + 6 * powXm1);
	clipEig		= 2 * (kIndex + yIndex * powX + 7 * powXm1);

	double2 CST;
	double2 TMP;
	double16 SIGA = (double16)(	data[clipOne+0],data[clipOne+1],	// s0, s1
								data[clipTwo+0],data[clipTwo+1],	// s2, s3
								data[clipThr+0],data[clipThr+1],	// s4, s5
								data[clipFou+0],data[clipFou+1],	// s6, s7
								data[clipFiv+0],data[clipFiv+1],	// s8, s9
								data[clipSix+0],data[clipSix+1],	// sa, sb
								data[clipSev+0],data[clipSev+1],	// sc, sd
								data[clipEig+0],data[clipEig+1]);	// se, sf

	
	int coeffUse = kIndex * (size / powX);	
	int red = size/4;
	double2 clSet1;

	int quad = coeffUse/red;
	int buad = coeffUse%red;
	switch(quad) {
		case 0: clSet1 = (double2)(	twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)(	twiddle[buad].y,	-twiddle[buad].x); break;
		case 2:	clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s8 * clSet1.x - SIGA.s9 * clSet1.y;
		TMP.y = SIGA.s8 * clSet1.y + SIGA.s9 * clSet1.x;
		SIGA.s8 = TMP.x;
		SIGA.s9 = TMP.y;
	}	

	quad = (2*coeffUse)/red;
	buad = (2*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TMP.y = SIGA.s4 * clSet1.y + SIGA.s5 * clSet1.x;
		SIGA.s4 = TMP.x;
		SIGA.s5 = TMP.y;
	}	

	quad = (3*coeffUse)/red;
	buad = (3*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.sc * clSet1.x - SIGA.sd * clSet1.y;
		TMP.y = SIGA.sc * clSet1.y + SIGA.sd * clSet1.x;
		SIGA.sc = TMP.x;
		SIGA.sd = TMP.y;
	}	

	quad = (4*coeffUse)/red;
	buad = (4*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s2 * clSet1.x - SIGA.s3 * clSet1.y;
		TMP.y = SIGA.s2 * clSet1.y + SIGA.s3 * clSet1.x;
		SIGA.s2 = TMP.x;
		SIGA.s3 = TMP.y;
	}

	quad = (5*coeffUse)/red;
	buad = (5*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.sa * clSet1.x - SIGA.sb * clSet1.y;
		TMP.y = SIGA.sa * clSet1.y + SIGA.sb * clSet1.x;
		SIGA.sa = TMP.x;
		SIGA.sb = TMP.y;
	}

	quad = (6*coeffUse)/red;
	buad = (6*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.s6 * clSet1.x - SIGA.s7 * clSet1.y;
		TMP.y = SIGA.s6 * clSet1.y + SIGA.s7 * clSet1.x;
		SIGA.s6 = TMP.x;
		SIGA.s7 = TMP.y;
	}	

	quad = (7*coeffUse)/red;
	buad = (7*coeffUse)%red;
	switch(quad) {
		case 0: clSet1 = (double2)( twiddle[buad].x, 	 twiddle[buad].y); break;
		case 1: clSet1 = (double2)( twiddle[buad].y,	-twiddle[buad].x); break;
		case 2: clSet1 = (double2)(-twiddle[buad].x,	-twiddle[buad].y); break;
		case 3: clSet1 = (double2)(-twiddle[buad].y,	 twiddle[buad].x); break;
	}
	if (dir == 0) clSet1.y *= -1;
	if (kIndex != 0) {
		TMP.x = SIGA.se * clSet1.x - SIGA.sf * clSet1.y;
		TMP.y = SIGA.se * clSet1.y + SIGA.sf * clSet1.x;
		SIGA.se = TMP.x;
		SIGA.sf = TMP.y;
	}	

	double16 SIGB = (double16)(	SIGA.s0 + SIGA.s2,
								SIGA.s1 + SIGA.s3,
								SIGA.s0 - SIGA.s2,
								SIGA.s1 - SIGA.s3,
								SIGA.s4 + SIGA.s6,
								SIGA.s5 + SIGA.s7,
								SIGA.s4 - SIGA.s6,
								SIGA.s5 - SIGA.s7,
								SIGA.s8 + SIGA.sa,
								SIGA.s9 + SIGA.sb,
								SIGA.s8 - SIGA.sa,
								SIGA.s9 - SIGA.sb,
								SIGA.sc + SIGA.se,
								SIGA.sd + SIGA.sf,
								SIGA.sc - SIGA.se,
								SIGA.sd - SIGA.sf);

	if (dir == 1) {
		TMP = (double2)((SIGB.sa + SIGB.sb)*d707, (SIGB.sf - SIGB.se)*d707);
		SIGB.sb = (SIGB.sb - SIGB.sa) * d707;
		SIGB.sf = (SIGB.sf + SIGB.se) * -d707;
		SIGB.sa = TMP.x;	
		SIGB.se = TMP.y;
		TMP.x	= SIGB.s7; SIGB.s7 = -SIGB.s6; SIGB.s6 = TMP.x; 
	}
	else if (dir == 0) {
		TMP = (double2)((SIGB.sa - SIGB.sb)*d707, (SIGB.sf + SIGB.se)*-d707);
		SIGB.sb = (SIGB.sb + SIGB.sa) * d707;
		SIGB.sf = (SIGB.se - SIGB.sf) * d707;
		SIGB.sa = TMP.x;
		SIGB.se = TMP.y;
		TMP.x 	= -SIGB.s7; SIGB.s7 = SIGB.s6; SIGB.s6 = TMP.x;
	}

	SIGA.s0 = SIGB.s0 + SIGB.s4;
	SIGA.s1 = SIGB.s1 + SIGB.s5;
	SIGA.s2 = SIGB.s2 + SIGB.s6;
	SIGA.s3 = SIGB.s3 + SIGB.s7;
	SIGA.s4 = SIGB.s0 - SIGB.s4;
	SIGA.s5 = SIGB.s1 - SIGB.s5;
	SIGA.s6 = SIGB.s2 - SIGB.s6;
	SIGA.s7 = SIGB.s3 - SIGB.s7;
	SIGA.s8 = SIGB.s8 + SIGB.sc;
	SIGA.s9 = SIGB.s9 + SIGB.sd;
	SIGA.sa = SIGB.sa + SIGB.se;
	SIGA.sb = SIGB.sb + SIGB.sf;
	if (dir == 1) {
		SIGA.sc = SIGB.s9 - SIGB.sd;
		SIGA.sd = SIGB.sc - SIGB.s8;
		SIGA.se = SIGB.sb - SIGB.sf;
		SIGA.sf = SIGB.se - SIGB.sa;
	}
	else if (dir == 0) {
		SIGA.sc = SIGB.sd - SIGB.s9;
		SIGA.sd = SIGB.s8 - SIGB.sc;
		SIGA.se = SIGB.sf - SIGB.sb;
		SIGA.sf = SIGB.sa - SIGB.se;
	}

	#if 1
	data[clipOne+0] = SIGA.s0 + SIGA.s8;
	data[clipOne+1] = SIGA.s1 + SIGA.s9;
	data[clipTwo+0] = SIGA.s2 + SIGA.sa;
	data[clipTwo+1] = SIGA.s3 + SIGA.sb;
	data[clipThr+0] = SIGA.s4 + SIGA.sc;
	data[clipThr+1] = SIGA.s5 + SIGA.sd;
	data[clipFou+0] = SIGA.s6 + SIGA.se;
	data[clipFou+1] = SIGA.s7 + SIGA.sf;
	data[clipFiv+0] = SIGA.s0 - SIGA.s8;
	data[clipFiv+1] = SIGA.s1 - SIGA.s9;
	data[clipSix+0] = SIGA.s2 - SIGA.sa;
	data[clipSix+1] = SIGA.s3 - SIGA.sb;
	data[clipSev+0] = SIGA.s4 - SIGA.sc;
	data[clipSev+1] = SIGA.s5 - SIGA.sd;
	data[clipEig+0] = SIGA.s6 - SIGA.se; 
	data[clipEig+1] = SIGA.s7 - SIGA.sf;
	#endif
	#if 0
	data[clipOne+0] = 0;//
	data[clipOne+1] = 0;
	data[clipTwo+0] = 0;
	data[clipTwo+1] = 0;
	data[clipThr+0] = 0;
	data[clipThr+1] = 0;
	data[clipFou+0] = 0;
	data[clipFou+1] = 0;
	data[clipFiv+0] = 0;//
	data[clipFiv+1] = 0;
	data[clipSix+0] = 0;
	data[clipSix+1] = 0;
	data[clipSev+0] = 0;
	data[clipSev+1] = 0;
	data[clipEig+0] = idZ;
	data[clipEig+1] = 0;
	#endif
	#if 0
	data[2*(idZ*x*y+idY*x+idX)] = powX;
	data[2*(idZ*x*y+idY*x+idX)+1] = 0;
	#endif
}


__kernel void DIT2C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int powX = topePow(2., stage, 11);
	int powXm1 = powX/2;

	int clipStart, clipEnd;
	int yIndex, kIndex;

	int BASE 	= 0;
	int STRIDE 	= 1;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}

	clipStart 	= 2*(BASE+STRIDE*(kIndex + yIndex * powX));
	clipEnd 	= 2*(BASE+STRIDE*(kIndex + yIndex * powX + powXm1));

	double2 clSet1;
	clSet1.x 	= cos(CLPT*kIndex/powX);
	clSet1.y 	= sin(CLPT*kIndex/powX);
	if (dir == 0) clSet1.y *= -1;

	double4 LOC = (double4)(	data[clipStart + 0],	data[clipStart + 1],
								data[clipEnd + 0],		data[clipEnd + 1]);

	double4 FIN = rad2(LOC,clSet1);

	data[clipStart + 0] = FIN.x;
	data[clipStart + 1] = FIN.y;
	data[clipEnd + 0] 	= FIN.z;
	data[clipEnd + 1] 	= FIN.w;
}

__kernel void DITRC2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type,
						unsigned int radix)
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int powX   = topePowInt(radix, stage);
	int powXm1 = powX/radix;

	int clipOne, clipTwo, clipThr;
	int yIndex, kIndex;

	int BASE   = 0;
	int STRIDE = 1;
	switch(type){
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}

	int clip[10];
	int i;
	for (i=0; i<radix; i++){
		clip[i] = 2 * (BASE+STRIDE*(kIndex + yIndex * powX + i * powXm1));
	}	
	double SIG[2*10];

	for(i=0; i<radix; i++){
		
		SIG[2*i]=data[clip[i]+0];
		SIG[2*i+1]=data[clip[i]+1];
	}
	
	
	double2 clSet;
	double2 TEMPC;

	if(kIndex != 0){

		for(i=1; i < radix; i++){
			
			clSet.x    =  cos( i * CLPT*kIndex/powX );
			clSet.y    = -sin( i * CLPT*kIndex/powX );
			if (dir == 0) clSet.y *=-1;
			TEMPC.x    = SIG[2*i] * clSet.x - SIG[2*i+1] * clSet.y;
			TEMPC.y    = SIG[2*i] * clSet.y + SIG[2*i+1] * clSet.x;
			SIG[2*i]   = TEMPC.x;
			SIG[2*i+1] = TEMPC.y;
		}

	}

	switch(radix){
		
		case 3:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5];
			data[clip[1]+0] = 
							SIG[0] + (-wa3*SIG[2] + wb3* SIG[3]) 
							+(-wa3*SIG[4] - wb3*SIG[5]);
			data[clip[1]+1] = 
							SIG[1] + (-wb3*SIG[2] - wa3 * SIG[3]) 
							+(wb3*SIG[4] - wa3*SIG[5]);
			data[clip[2]+0] = 
							SIG[0] + (-wa3*SIG[2] - wb3* SIG[3]) 
							+(-wa3*SIG[4] + wb3*SIG[5]);
			data[clip[2]+1] = 
							SIG[1] + (wb3*SIG[2] - wa3* SIG[3]) 
							+(-wb3*SIG[4] - wa3*SIG[5]);
			break;
		case 5:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] + SIG[8];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] + SIG[9];
			data[clip[1]+0] = SIG[0] + (wa5*SIG[2] + wb5*SIG[3]) 
							+(-wc5*SIG[4] + wd5*SIG[5]) + (-wc5*SIG[6] 
							- wd5*SIG[7])+(wa5*SIG[8] - 	wb5*SIG[9]);
			data[clip[1]+1] = SIG[1] + (wa5*SIG[3] - wb5*SIG[2])
							+(-wc5*SIG[5] - wd5*SIG[4]) + (-wc5*SIG[7] +
							wd5*SIG[6])+(wa5*SIG[9] + wb5*SIG[8]);
			data[clip[2]+0] = SIG[0] + (-wc5*SIG[2] + wd5*SIG[3])
							+(wa5*SIG[4] - wb5*SIG[5]) + (wa5*SIG[6] +
							wb5*SIG[7])+(-wc5*SIG[8] - wd5*SIG[9]); 
			data[clip[2]+1] = SIG[1] + (-wc5*SIG[3] - wd5*SIG[2]) 
							+(wa5*SIG[5] + wb5*SIG[4]) + (wa5*SIG[7] 
							- wb5*SIG[6])+(-wc5*SIG[9] +	wd5*SIG[8]);
			data[clip[3]+0] = SIG[0] + (-wc5*SIG[2] - wd5*SIG[3]) 
							+(wa5*SIG[4] + wb5*SIG[5]) + (wa5*SIG[6] 
							- wb5*SIG[7])+(-wc5*SIG[8] + wd5*SIG[9]);
			data[clip[3]+1] = SIG[1] + (-wc5*SIG[3] + wd5*SIG[2]) 
							+(wa5*SIG[5] - wb5*SIG[4]) + (wa5*SIG[7] 
							+ wb5*SIG[6])+(-wc5*SIG[9] - wd5*SIG[8]);
			data[clip[4]+0] = SIG[0] + (wa5*SIG[2] - wb5*SIG[3]) 
							+(-wc5*SIG[4] - wd5*SIG[5]) + (-wc5*SIG[6] 
							+ wd5*SIG[7])+(wa5*SIG[8] + 	wb5*SIG[9]);
			data[clip[4]+1] = SIG[1] + (wa5*SIG[3] + wb5*SIG[2]) 
							+(-wc5*SIG[5] + wd5*SIG[4]) + (-wc5*SIG[7] 
							- wd5*SIG[6])+(wa5*SIG[9] - wb5*SIG[8]);
			break;
		case 6:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] + SIG[8] + SIG[10];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] + SIG[9] + SIG[11];

			data[clip[1]+0] = SIG[0] + (wa6*SIG[2] + wb6*SIG[3]) +
							(-wa6*SIG[4] + wb6*SIG[5]) + (-SIG[6])+
							(-wa6*SIG[8] - wb6*SIG[9]) + (wa6*SIG[10] - wb6*SIG[11]); 
			data[clip[1]+1] = SIG[1] + (wa6*SIG[3] - wb6*SIG[2]) +
							(-wa6*SIG[5] - wb6*SIG[4]) + (-SIG[7])+
							(-wa6*SIG[9] +	wb6*SIG[8]) + (wa6*SIG[11] + wb6*SIG[10]);

			data[clip[2]+0] = SIG[0] + (-wa6*SIG[2] + wb6*SIG[3]) +
							(-wa6*SIG[4] - wb6*SIG[5]) + (SIG[6])+
							(-wa6*SIG[8] + wb6*SIG[9]) + (-wa6*SIG[10] - wb6*SIG[11]); 
			data[clip[2]+1] = SIG[1] + (-wa6*SIG[3] - wb6*SIG[2]) +
							(-wa6*SIG[5] + wb6*SIG[4]) + (SIG[7])+
							(-wa6*SIG[9] -	wb6*SIG[8]) + (-wa6*SIG[11] + wb6*SIG[10]);

			data[clip[3]+0] = SIG[0] + (-SIG[2]) + (SIG[4]) + (-SIG[6]) +
							(SIG[8]) + (-SIG[10]);
			data[clip[3]+1] = SIG[1] + (-SIG[3]) + (SIG[5]) + (-SIG[7]) +
							(SIG[9]) + (-SIG[11]);

			data[clip[4]+0] = SIG[0] + (-wa6*SIG[2] - wb6*SIG[3]) +
							(-wa6*SIG[4] + wb6*SIG[5]) + (SIG[6])+
							(-wa6*SIG[8] - wb6*SIG[9]) + (-wa6*SIG[10] + wb6*SIG[11]); 

			data[clip[4]+1] = SIG[1] + (-wa6*SIG[3] + wb6*SIG[2]) +
							(-wa6*SIG[5] - wb6*SIG[4]) + (SIG[7])+
							(-wa6*SIG[9] +	wb6*SIG[8]) + (-wa6*SIG[11] - wb6*SIG[10]);

			data[clip[5]+0] = SIG[0] + (wa6*SIG[2] - wb6*SIG[3]) +
							(-wa6*SIG[4] - wb6*SIG[5]) + (-SIG[6])+
							(-wa6*SIG[8] + wb6*SIG[9]) + (wa6*SIG[10] + wb6*SIG[11]); 

			data[clip[5]+1] =  SIG[1] + (wa6*SIG[3] + wb6*SIG[2]) +
							(-wa6*SIG[5] + wb6*SIG[4]) + (-SIG[7])+
							(-wa6*SIG[9] -	wb6*SIG[8]) + (wa6*SIG[11] - wb6*SIG[10]);
		
			break;
		case 7:	
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] 
							+ SIG[8] + SIG[10] + SIG[12];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] 
							+ SIG[9] + SIG[11] + SIG[13];
			data[clip[1]+0] = SIG[0] + (wa7*SIG[2] + wb7*SIG[3]) 
							+(-wc7*SIG[4] + wd7*SIG[5]) 
							+ (-we7*SIG[6] + wf7*SIG[7]) 
							+ (-we7*SIG[8] - wf7*SIG[9]) 
							+ (-wc7*SIG[10] - wd7*SIG[11]) 
							+ (wa7*SIG[12] - wb7*SIG[13]);
			data[clip[1]+1] = SIG[1] + (wa7*SIG[3] - wb7*SIG[2]) 
							+(-wc7*SIG[5] - wd7*SIG[4]) 
							+ (-we7*SIG[7] - wf7*SIG[6]) 
							+ (-we7*SIG[9] + wf7*SIG[8]) 
							+ (-wc7*SIG[11] + wd7*SIG[10]) 
							+ (wa7*SIG[13] + wb7*SIG[12]);
			data[clip[2]+0] = SIG[0] + (-wc7*SIG[2] + wd7*SIG[3]) 
							+(-we7*SIG[4] - wf7*SIG[5]) 
							+ (wa7*SIG[6] - wb7*SIG[7]) 
							+ (wa7*SIG[8] + wb7*SIG[9]) 
							+ (-we7*SIG[10] + wf7*SIG[11]) 
							+ (-wc7*SIG[12] - wd7*SIG[13]);
			data[clip[2]+1] = SIG[1] + (-wc7*SIG[3] - wd7*SIG[2]) 
							+(-we7*SIG[5] + wf7*SIG[4]) 
							+ (wa7*SIG[7] + wb7*SIG[6]) 
							+ (wa7*SIG[9] - wb7*SIG[8]) 
							+ (-we7*SIG[11] - wf7*SIG[10]) 
							+ (-wc7*SIG[13] + wd7*SIG[12]);
			data[clip[3]+0] = SIG[0] + (-we7*SIG[2] + wf7*SIG[3]) 
							+(wa7*SIG[4] - wb7*SIG[5]) 
							+ (-wc7*SIG[6] + wd7*SIG[7]) 
							+ (-wc7*SIG[8] - wd7*SIG[9]) 
							+ (wa7*SIG[10] + wb7*SIG[11]) 
							+ (-we7*SIG[12] - wf7*SIG[13]);
			data[clip[3]+1] = SIG[1] + (-we7*SIG[3] - wf7*SIG[2]) 
							+(wa7*SIG[5] + wb7*SIG[4]) 
							+ (-wc7*SIG[7] - wd7*SIG[6]) 
							+ (-wc7*SIG[9] + wd7*SIG[8]) 
							+ (wa7*SIG[11] - wb7*SIG[10]) 
							+ (-we7*SIG[13] + wf7*SIG[12]);
			data[clip[4]+0] = SIG[0] + (-we7*SIG[2] - wf7*SIG[3]) 
							+(wa7*SIG[4] + wb7*SIG[5]) 
							+ (-wc7*SIG[6] - wd7*SIG[7]) 
							+ (-wc7*SIG[8] + wd7*SIG[9]) 
							+ (wa7*SIG[10] - wb7*SIG[11]) 
							+ (-we7*SIG[12] + wf7*SIG[13]);
			data[clip[4]+1] = SIG[1] + (-we7*SIG[3] + wf7*SIG[2]) 
							+(wa7*SIG[5] - wb7*SIG[4]) 
							+ (-wc7*SIG[7] + wd7*SIG[6]) 
							+ (-wc7*SIG[9] - wd7*SIG[8]) 
							+ (wa7*SIG[11] + wb7*SIG[10]) 
							+ (-we7*SIG[13] - wf7*SIG[12]);
			data[clip[5]+0] = SIG[0] + (-wc7*SIG[2] - wd7*SIG[3]) 
							+(-we7*SIG[4] + wf7*SIG[5]) 
							+ (wa7*SIG[6] + wb7*SIG[7]) 
							+ (wa7*SIG[8] - wb7*SIG[9]) 
							+ (-we7*SIG[10] - wf7*SIG[11]) 
							+ (-wc7*SIG[12] + wd7*SIG[13]);
			data[clip[5]+1] = SIG[1] + (-wc7*SIG[3] + wd7*SIG[2]) 
							+(-we7*SIG[5] - wf7*SIG[4]) 
							+ (wa7*SIG[7] - wb7*SIG[6]) 
							+ (wa7*SIG[9] + wb7*SIG[8]) 
							+ (-we7*SIG[11] + wf7*SIG[10]) 
							+ (-wc7*SIG[13] - wd7*SIG[12]);
			data[clip[6]+0] = SIG[0] + (wa7*SIG[2] - wb7*SIG[3]) 
							+(-wc7*SIG[4] - wd7*SIG[5]) 
							+ (-we7*SIG[6] - wf7*SIG[7]) 
							+ (-we7*SIG[8] + wf7*SIG[9]) 
							+ (-wc7*SIG[10] + wd7*SIG[11]) 
							+ (wa7*SIG[12] + wb7*SIG[13]);
			data[clip[6]+1] = SIG[1] + (wa7*SIG[3] + wb7*SIG[2]) 
							+(-wc7*SIG[5] + wd7*SIG[4]) 
							+ (-we7*SIG[7] + wf7*SIG[6]) 
							+ (-we7*SIG[9] - wf7*SIG[8]) 
							+ (-wc7*SIG[11] - wd7*SIG[10]) 
							+ (wa7*SIG[13] - wb7*SIG[12]);
			break;
		case 10:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] + SIG[8]
							+ SIG[10] + SIG[12] + SIG[14] + SIG[16] + SIG[18];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] + SIG[9]
							+ SIG[11] + SIG[13] + SIG[15] + SIG[17] + SIG[19];
	
			data[clip[1]+0] = SIG[0] + (wa10*SIG[2] + wb10*SIG[3])
							+ (wc10*SIG[4] + wd10*SIG[5]) + (-wc10*SIG[6]
							+ wd10*SIG[7]) + (-wa10*SIG[8] + wb10*SIG[9])
							+ (-SIG[10]) + (-wa10*SIG[12] - wb10*SIG[13])
							+ (-wc10*SIG[14] - wd10*SIG[15]) + (wc10*SIG[16]
							- wd10*SIG[17]) + (wa10*SIG[18] - wb10*SIG[19]);
							
			data[clip[1]+1]  = SIG[1] + (wa10*SIG[3] - wb10*SIG[2])
							+(wc10*SIG[5] - wd10*SIG[4]) + (-wc10*SIG[7]
							- wd10*SIG[6]) + (-wa10*SIG[9] - wb10*SIG[8])
							+ (-SIG[11]) + (-wa10*SIG[13] + wb10*SIG[12])
							+ (-wc10*SIG[15] + wd10*SIG[14]) + (wc10*SIG[17]
							+ wd10*SIG[16]) +(wa10*SIG[19] + wb10*SIG[18]) ;

			data[clip[2]+0] = SIG[0] + (wc10*SIG[2] + wd10*SIG[3])
							+(-wa10*SIG[4] + wb10*SIG[5]) + (-wa10*SIG[6]
							- wb10*SIG[7]) + (wc10*SIG[8] - wd10*SIG[9])
							+ (SIG[10]) + (wc10*SIG[12] + 	wd10*SIG[13])
							+ (-wa10*SIG[14] + wb10*SIG[15]) + (-wa10*SIG[16]
							- wb10*SIG[17]) + (wc10*SIG[18] - wd10*SIG[19]);
			data[clip[2]+1] = SIG[1] + (wc10*SIG[3] - wd10*SIG[2])
							+(-wa10*SIG[5] - wb10*SIG[4]) + (-wa10*SIG[7]
							+ wb10*SIG[6]) + (wc10*SIG[9] + wd10*SIG[8])
							+ (SIG[11]) + (wc10*SIG[13] - wd10*SIG[12])
							+ (-wa10*SIG[15] - wb10*SIG[14]) + (-wa10*SIG[17]
							+ wb10*SIG[16]) +(wc10*SIG[19] + wd10*SIG[18]) ;


			data[clip[3]+0] = SIG[0] + (-wc10*SIG[2] + wd10*SIG[3]) 
							+(-wa10*SIG[4] - wb10*SIG[5]) + (wa10*SIG[6]
							- wb10*SIG[7]) + (wc10*SIG[8] + wd10*SIG[9]) 
							+ (-SIG[10]) + (wc10*SIG[12] - 	wd10*SIG[13]) 
							+ (wa10*SIG[14] + wb10*SIG[15]) + (-wa10*SIG[16]
							+ wb10*SIG[17]) + (-wc10*SIG[18] - wd10*SIG[19]);
			data[clip[3]+1] = SIG[1] + (-wc10*SIG[3] - wd10*SIG[2]) 
							+(-wa10*SIG[5] + wb10*SIG[4]) + (wa10*SIG[7] 
							+ wb10*SIG[6]) + (wc10*SIG[9] - wd10*SIG[8])
							+ (-SIG[11]) + (wc10*SIG[13] + wd10*SIG[12]) 
							+ (wa10*SIG[15] - wb10*SIG[14]) + (-wa10*SIG[17]
							- wb10*SIG[16]) +(-wc10*SIG[19] + wd10*SIG[18]) ;

			data[clip[4]+0] = SIG[0] + (-wa10*SIG[2] + wb10*SIG[3]) 
							+ (wc10*SIG[4] - wd10*SIG[5]) + (wc10*SIG[6]
							+ wd10*SIG[7]) + (-wa10*SIG[8] - wb10*SIG[9]) 
							+ (SIG[10]) + (-wa10*SIG[12] + 	wb10*SIG[13]) 
							+ (wc10*SIG[14] - wd10*SIG[15]) + (wc10*SIG[16]
							+ wd10*SIG[17]) + (-wa10*SIG[18] - wb10*SIG[19]);
			data[clip[4]+1]	= SIG[1] + (-wa10*SIG[3] - wb10*SIG[2]) 
							+ (wc10*SIG[5] + wd10*SIG[4]) + (wc10*SIG[7]
							- wd10*SIG[6]) + (-wa10*SIG[9] + wb10*SIG[8])
							+ (SIG[11]) + (-wa10*SIG[13] - wb10*SIG[12]) 
							+ (wc10*SIG[15] + wd10*SIG[14])	+ (wc10*SIG[17]
							- wd10*SIG[16]) + (-wa10*SIG[19] + wb10*SIG[18]) ;

			data[clip[5]+0] = SIG[0] + (-SIG[2]) + SIG[4] + (-SIG[6])
							+ SIG[8] + (-SIG[10]) + SIG[12] + (-SIG[14])
							+ SIG[16] + (-SIG[18]);
			data[clip[5]+1] = SIG[1] + (-SIG[3]) + SIG[5] + (-SIG[7])
							+ SIG[9] + (-SIG[11]) + SIG[13] + (-SIG[15])
							+ SIG[17] + (-SIG[19]);
	
			data[clip[6]+0] = SIG[0] + (-wa10*SIG[2] - wb10*SIG[3])
							+(wc10*SIG[4] + wd10*SIG[5]) + (wc10*SIG[6]
							- wd10*SIG[7]) + (-wa10*SIG[8] + wb10*SIG[9])
							+ (SIG[10]) + (-wa10*SIG[12] - wb10*SIG[13])
							+ (wc10*SIG[14] + wd10*SIG[15]) + (wc10*SIG[16]
							- wd10*SIG[17]) + (-wa10*SIG[18] + wb10*SIG[19]);
			data[clip[6]+1] = SIG[1] + (-wa10*SIG[3] + wb10*SIG[2])
							+(wc10*SIG[5] - wd10*SIG[4]) + (wc10*SIG[7]
							+ wd10*SIG[6]) + (-wa10*SIG[9] - wb10*SIG[8])
							+ (SIG[11]) + (-wa10*SIG[13] + wb10*SIG[12])
							+ (wc10*SIG[15] - wd10*SIG[14]) + (wc10*SIG[17]
							+ wd10*SIG[16]) +(-wa10*SIG[19] - wb10*SIG[18]) ;


			data[clip[7]+0] = SIG[0] + (-wc10*SIG[2] - wd10*SIG[3])
							+(-wa10*SIG[4] + wb10*SIG[5]) + (wa10*SIG[6]
							+ wb10*SIG[7]) + (wc10*SIG[8] - wd10*SIG[9])
							+ (-SIG[10]) + (wc10*SIG[12] + wd10*SIG[13])
							+ (wa10*SIG[14] - wb10*SIG[15]) + (-wa10*SIG[16]
							- wb10*SIG[17]) + (-wc10*SIG[18] + wd10*SIG[19]);
			data[clip[7]+1] = SIG[1] + (-wc10*SIG[3] + wd10*SIG[2])
							+(-wa10*SIG[5] -wb10*SIG[4]) + (wa10*SIG[7]
							- wb10*SIG[6]) + (wc10*SIG[9] + wd10*SIG[8])
							+ (-SIG[11]) + (wc10*SIG[13] - wd10*SIG[12])
							+ (wa10*SIG[15] + wb10*SIG[14]) + (-wa10*SIG[17]
							+ wb10*SIG[16]) +(-wc10*SIG[19] - wd10*SIG[18]) ;


			data[clip[8]+0] = SIG[0] + (wc10*SIG[2] - wd10*SIG[3])
							+(-wa10*SIG[4] - wb10*SIG[5]) + (-wa10*SIG[6]
							+ wb10*SIG[7]) + (wc10*SIG[8] + wd10*SIG[9])
							+ (SIG[10]) + (wc10*SIG[12] - wd10*SIG[13])
							+ (-wa10*SIG[14] - wb10*SIG[15]) + (-wa10*SIG[16]
							+ wb10*SIG[17]) + (wc10*SIG[18] + wd10*SIG[19]);
			data[clip[8]+1] = SIG[1] + (wc10*SIG[3] + wd10*SIG[2])
							+(-wa10*SIG[5] + wb10*SIG[4]) + (-wa10*SIG[7]
							- wb10*SIG[6]) + (wc10*SIG[9] - wd10*SIG[8])
							+ (SIG[11]) + (wc10*SIG[13] + wd10*SIG[12])
							+ (-wa10*SIG[15] + wb10*SIG[14]) + (-wa10*SIG[17]
							- wb10*SIG[16]) +(wc10*SIG[19] - wd10*SIG[18]) ;


			data[clip[9]+0] = SIG[0] + (wa10*SIG[2] - wb10*SIG[3])
							+(wc10*SIG[4] - wd10*SIG[5]) + (-wc10*SIG[6]
							- wd10*SIG[7]) + (-wa10*SIG[8] - wb10*SIG[9])
							+ (-SIG[10]) + (-wa10*SIG[12] + wb10*SIG[13])
							+ (-wc10*SIG[14] + wd10*SIG[15]) + (wc10*SIG[16]
							+ wd10*SIG[17]) + (wa10*SIG[18] + wb10*SIG[19]);
			data[clip[9]+1] = SIG[1] + (wa10*SIG[3] + wb10*SIG[2])
							+(wc10*SIG[5] + wd10*SIG[4]) + (-wc10*SIG[7]
							+ wd10*SIG[6]) + (-wa10*SIG[9] + wb10*SIG[8])
							+ (-SIG[11]) + (-wa10*SIG[13] - wb10*SIG[12])
							+ (-wc10*SIG[15] - wd10*SIG[14]) + (wc10*SIG[17]
							- wd10*SIG[16]) +(wa10*SIG[19] - wb10*SIG[18]) ;
			break;
		}

	}


__kernel void DIT7C2C(
			__global double *data,
			const int size,
			unsigned int stage,
			unsigned int dir)
{
#if 1 // correct
	int idX = get_global_id(0);

//	int powX = topePow(7.,stage,11);
	int powX = topePowInt(7,stage);
	int powXm1 = powX/7;
	// x =pow(5.0,2)=25 whereas x=pow(5.0f,2)=24	
	//
	int clipOne, clipTwo, clipThr, clipFou, clipFiv, clipSix, clipSev;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);
	clipFiv		= 2 * (kIndex + yIndex * powX + 4 * powXm1);
	clipSix		= 2 * (kIndex + yIndex * powX + 5 * powXm1);
	clipSev		= 2 * (kIndex + yIndex * powX + 6 * powXm1);
	
	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
	double2 SIG6A = (double2)(	data[clipSix+0],data[clipSix+1]);
	double2 SIG7A = (double2)(	data[clipSev+0],data[clipSev+1]);

	double2 clSet2, clSet3, clSet4, clSet5, clSet6, clSet7, temp2;
	
	if (kIndex!=0) {
		
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
	
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;

		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
	
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;

		clSet6.x = cos(5*CLPT*kIndex/powX);
		clSet6.y = -sin(5*CLPT*kIndex/powX);
		if (dir == 0) clSet6.y *= -1;
		TEMPC.x = SIG6A.x * clSet6.x - SIG6A.y * clSet6.y;
		TEMPC.y = SIG6A.x * clSet6.y + SIG6A.y * clSet6.x;
		SIG6A.x = TEMPC.x;
		SIG6A.y = TEMPC.y;

		clSet7.x = cos(6*CLPT*kIndex/powX);
		clSet7.y = -sin(6*CLPT*kIndex/powX);
		if (dir == 0) clSet7.y *= -1;
		TEMPC.x = SIG7A.x * clSet7.x - SIG7A.y * clSet7.y;
		TEMPC.y = SIG7A.x * clSet7.y + SIG7A.y * clSet7.x;
		SIG7A.x = TEMPC.x;
		SIG7A.y = TEMPC.y;
	}	
	
	data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 
					+ SIG5A.x + SIG6A.x + SIG7A.x;
	data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 
					+ SIG5A.y + SIG6A.y + SIG7A.y;
	data[clipTwo+0] = SIG4A.s0 + (wa7*SIG4A.s2 + wb7*SIG4A.s3) 
					+(-wc7*SIG4A.s4 + wd7*SIG4A.s5) 
					+ (-we7*SIG4A.s6 + wf7*SIG4A.s7) 
					+ (-we7*SIG5A.x - wf7*SIG5A.y) 
					+ (-wc7*SIG6A.x - wd7*SIG6A.y) 
					+ (wa7*SIG7A.x - wb7*SIG7A.y);
	data[clipTwo+1] = SIG4A.s1 + (wa7*SIG4A.s3 - wb7*SIG4A.s2) 
					+(-wc7*SIG4A.s5 - wd7*SIG4A.s4) 
					+ (-we7*SIG4A.s7 - wf7*SIG4A.s6) 
					+ (-we7*SIG5A.y + wf7*SIG5A.x) 
					+ (-wc7*SIG6A.y + wd7*SIG6A.x) 
					+ (wa7*SIG7A.y + wb7*SIG7A.x);
	data[clipThr+0] = SIG4A.s0 + (-wc7*SIG4A.s2 + wd7*SIG4A.s3) 
					+(-we7*SIG4A.s4 - wf7*SIG4A.s5) 
					+ (wa7*SIG4A.s6 - wb7*SIG4A.s7) 
					+ (wa7*SIG5A.x + wb7*SIG5A.y) 
					+ (-we7*SIG6A.x + wf7*SIG6A.y) 
					+ (-wc7*SIG7A.x - wd7*SIG7A.y);
	data[clipThr+1] = SIG4A.s1 + (-wc7*SIG4A.s3 - wd7*SIG4A.s2) 
					+(-we7*SIG4A.s5 + wf7*SIG4A.s4) 
					+ (wa7*SIG4A.s7 + wb7*SIG4A.s6) 
					+ (wa7*SIG5A.y - wb7*SIG5A.x) 
					+ (-we7*SIG6A.y - wf7*SIG6A.x) 
					+ (-wc7*SIG7A.y + wd7*SIG7A.x);
	data[clipFou+0] = SIG4A.s0 + (-we7*SIG4A.s2 + wf7*SIG4A.s3) 
					+(wa7*SIG4A.s4 - wb7*SIG4A.s5) 
					+ (-wc7*SIG4A.s6 + wd7*SIG4A.s7) 
					+ (-wc7*SIG5A.x - wd7*SIG5A.y) 
					+ (wa7*SIG6A.x + wb7*SIG6A.y) 
					+ (-we7*SIG7A.x - wf7*SIG7A.y);
	data[clipFou+1] = SIG4A.s1 + (-we7*SIG4A.s3 - wf7*SIG4A.s2) 
					+(wa7*SIG4A.s5 + wb7*SIG4A.s4) 
					+ (-wc7*SIG4A.s7 - wd7*SIG4A.s6) 
					+ (-wc7*SIG5A.y + wd7*SIG5A.x) 
					+ (wa7*SIG6A.y - wb7*SIG6A.x) 
					+ (-we7*SIG7A.y + wf7*SIG7A.x);
	data[clipFiv+0] = SIG4A.s0 + (-we7*SIG4A.s2 - wf7*SIG4A.s3) 
					+(wa7*SIG4A.s4 + wb7*SIG4A.s5) 
					+ (-wc7*SIG4A.s6 - wd7*SIG4A.s7) 
					+ (-wc7*SIG5A.x + wd7*SIG5A.y) 
					+ (wa7*SIG6A.x - wb7*SIG6A.y) 
					+ (-we7*SIG7A.x + wf7*SIG7A.y);
	data[clipFiv+1] = SIG4A.s1 + (-we7*SIG4A.s3 + wf7*SIG4A.s2) 
					+(wa7*SIG4A.s5 - wb7*SIG4A.s4) 
					+ (-wc7*SIG4A.s7 + wd7*SIG4A.s6) 
					+ (-wc7*SIG5A.y - wd7*SIG5A.x) 
					+ (wa7*SIG6A.y + wb7*SIG6A.x) 
					+ (-we7*SIG7A.y - wf7*SIG7A.x);
	data[clipSix+0] = SIG4A.s0 + (-wc7*SIG4A.s2 - wd7*SIG4A.s3) 
					+(-we7*SIG4A.s4 + wf7*SIG4A.s5) 
					+ (wa7*SIG4A.s6 + wb7*SIG4A.s7) 
					+ (wa7*SIG5A.x - wb7*SIG5A.y) 
					+ (-we7*SIG6A.x - wf7*SIG6A.y) 
					+ (-wc7*SIG7A.x + wd7*SIG7A.y);
	data[clipSix+1] = SIG4A.s1 + (-wc7*SIG4A.s3 + wd7*SIG4A.s2) 
					+(-we7*SIG4A.s5 - wf7*SIG4A.s4) 
					+ (wa7*SIG4A.s7 - wb7*SIG4A.s6) 
					+ (wa7*SIG5A.y + wb7*SIG5A.x) 
					+ (-we7*SIG6A.y + wf7*SIG6A.x) 
					+ (-wc7*SIG7A.y - wd7*SIG7A.x);
	data[clipSev+0] = SIG4A.s0 + (wa7*SIG4A.s2 - wb7*SIG4A.s3) 
					+(-wc7*SIG4A.s4 - wd7*SIG4A.s5) 
					+ (-we7*SIG4A.s6 - wf7*SIG4A.s7) 
					+ (-we7*SIG5A.x + wf7*SIG5A.y) 
					+ (-wc7*SIG6A.x + wd7*SIG6A.y) 
					+ (wa7*SIG7A.x + wb7*SIG7A.y);
	data[clipSev+1] = SIG4A.s1 + (wa7*SIG4A.s3 + wb7*SIG4A.s2) 
					+(-wc7*SIG4A.s5 + wd7*SIG4A.s4) 
					+ (-we7*SIG4A.s7 + wf7*SIG4A.s6) 
					+ (-we7*SIG5A.y - wf7*SIG5A.x) 
					+ (-wc7*SIG6A.y - wd7*SIG6A.x) 
					+ (wa7*SIG7A.y - wb7*SIG7A.x);
	#if 0 // Debug code
	data[clipOne+0] = 11;//kIndex;
	data[clipOne+1] = 111;//yIndex;
	data[clipTwo+0] = 22;//kIndex;
	data[clipTwo+1] = 222;//yIndex;
	data[clipThr+0] = 33;//kIndex;
	data[clipThr+1] = 333;//yIndex;
	data[clipFou+0] = 44;//kIndex;
	data[clipFou+1] = 444;//yIndex;
	data[clipFiv+0] = 55;//kIndex;
	data[clipFiv+1] = 555;//yIndex;
	#endif
#endif //end correct
}

__kernel void DITRC2C(
			__global double *data, 
			const int size,
			unsigned int stage,
			unsigned int dir, int radix)
{
	
	int idX=get_global_id(0);
	int powX = topePowInt(radix,stage);
	int powXm1 = powX/radix;

	int clip[10];
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;

	int i;
	for (i=0; i<radix; i++){
		clip[i] = 2 * (kIndex + yIndex * powX + i *powXm1);
	}

	
	double SIG[2*10];

	for(i=0; i<radix; i++){
		
		SIG[2*i]=data[clip[i]+0];
		SIG[2*i+1]=data[clip[i]+1];
	}
	
	
	double2 clSet;
	double2 TEMPC;

	if(kIndex != 0){

		for(i=1; i < radix; i++){
			
			clSet.x    =  cos( i * CLPT*kIndex/powX );
			clSet.y    = -sin( i * CLPT*kIndex/powX );
			if (dir == 0) clSet.y *=-1;
			TEMPC.x    = SIG[2*i] * clSet.x - SIG[2*i+1] * clSet.y;
			TEMPC.y    = SIG[2*i] * clSet.y + SIG[2*i+1] * clSet.x;
			SIG[2*i]   = TEMPC.x;
			SIG[2*i+1] = TEMPC.y;
		}

	}

	switch(radix){
		
		case 3:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5];
			data[clip[1]+0] = 
							SIG[0] + (-wa3*SIG[2] + wb3* SIG[3]) 
							+(-wa3*SIG[4] - wb3*SIG[5]);
			data[clip[1]+1] = 
							SIG[1] + (-wb3*SIG[2] - wa3 * SIG[3]) 
							+(wb3*SIG[4] - wa3*SIG[5]);
			data[clip[2]+0] = 
							SIG[0] + (-wa3*SIG[2] - wb3* SIG[3]) 
							+(-wa3*SIG[4] + wb3*SIG[5]);
			data[clip[2]+1] = 
							SIG[1] + (wb3*SIG[2] - wa3* SIG[3]) 
							+(-wb3*SIG[4] - wa3*SIG[5]);
			break;
		case 5:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] + SIG[8];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] + SIG[9];
			data[clip[1]+0] = SIG[0] + (wa5*SIG[2] + wb5*SIG[3]) 
							+(-wc5*SIG[4] + wd5*SIG[5]) + (-wc5*SIG[6] 
							- wd5*SIG[7])+(wa5*SIG[8] - 	wb5*SIG[9]);
			data[clip[1]+1] = SIG[1] + (wa5*SIG[3] - wb5*SIG[2])
							+(-wc5*SIG[5] - wd5*SIG[4]) + (-wc5*SIG[7] +
							wd5*SIG[6])+(wa5*SIG[9] + wb5*SIG[8]);
			data[clip[2]+0] = SIG[0] + (-wc5*SIG[2] + wd5*SIG[3])
							+(wa5*SIG[4] - wb5*SIG[5]) + (wa5*SIG[6] +
							wb5*SIG[7])+(-wc5*SIG[8] - wd5*SIG[9]); 
			data[clip[2]+1] = SIG[1] + (-wc5*SIG[3] - wd5*SIG[2]) 
							+(wa5*SIG[5] + wb5*SIG[4]) + (wa5*SIG[7] 
							- wb5*SIG[6])+(-wc5*SIG[9] +	wd5*SIG[8]);
			data[clip[3]+0] = SIG[0] + (-wc5*SIG[2] - wd5*SIG[3]) 
							+(wa5*SIG[4] + wb5*SIG[5]) + (wa5*SIG[6] 
							- wb5*SIG[7])+(-wc5*SIG[8] + wd5*SIG[9]);
			data[clip[3]+1] = SIG[1] + (-wc5*SIG[3] + wd5*SIG[2]) 
							+(wa5*SIG[5] - wb5*SIG[4]) + (wa5*SIG[7] 
							+ wb5*SIG[6])+(-wc5*SIG[9] - wd5*SIG[8]);
			data[clip[4]+0] = SIG[0] + (wa5*SIG[2] - wb5*SIG[3]) 
							+(-wc5*SIG[4] - wd5*SIG[5]) + (-wc5*SIG[6] 
							+ wd5*SIG[7])+(wa5*SIG[8] + 	wb5*SIG[9]);
			data[clip[4]+1] = SIG[1] + (wa5*SIG[3] + wb5*SIG[2]) 
							+(-wc5*SIG[5] + wd5*SIG[4]) + (-wc5*SIG[7] 
							- wd5*SIG[6])+(wa5*SIG[9] - wb5*SIG[8]);
			break;
		case 6:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] + SIG[8] + SIG[10];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] + SIG[9] + SIG[11];

			data[clip[1]+0] = SIG[0] + (wa6*SIG[2] + wb6*SIG[3]) +
							(-wa6*SIG[4] + wb6*SIG[5]) + (-SIG[6])+
							(-wa6*SIG[8] - wb6*SIG[9]) + (wa6*SIG[10] - wb6*SIG[11]); 
			data[clip[1]+1] = SIG[1] + (wa6*SIG[3] - wb6*SIG[2]) +
							(-wa6*SIG[5] - wb6*SIG[4]) + (-SIG[7])+
							(-wa6*SIG[9] +	wb6*SIG[8]) + (wa6*SIG[11] + wb6*SIG[10]);

			data[clip[2]+0] = SIG[0] + (-wa6*SIG[2] + wb6*SIG[3]) +
							(-wa6*SIG[4] - wb6*SIG[5]) + (SIG[6])+
							(-wa6*SIG[8] + wb6*SIG[9]) + (-wa6*SIG[10] - wb6*SIG[11]); 
			data[clip[2]+1] = SIG[1] + (-wa6*SIG[3] - wb6*SIG[2]) +
							(-wa6*SIG[5] + wb6*SIG[4]) + (SIG[7])+
							(-wa6*SIG[9] -	wb6*SIG[8]) + (-wa6*SIG[11] + wb6*SIG[10]);

			data[clip[3]+0] = SIG[0] + (-SIG[2]) + (SIG[4]) + (-SIG[6]) +
							(SIG[8]) + (-SIG[10]);
			data[clip[3]+1] = SIG[1] + (-SIG[3]) + (SIG[5]) + (-SIG[7]) +
							(SIG[9]) + (-SIG[11]);

			data[clip[4]+0] = SIG[0] + (-wa6*SIG[2] - wb6*SIG[3]) +
							(-wa6*SIG[4] + wb6*SIG[5]) + (SIG[6])+
							(-wa6*SIG[8] - wb6*SIG[9]) + (-wa6*SIG[10] + wb6*SIG[11]); 

			data[clip[4]+1] = SIG[1] + (-wa6*SIG[3] + wb6*SIG[2]) +
							(-wa6*SIG[5] - wb6*SIG[4]) + (SIG[7])+
							(-wa6*SIG[9] +	wb6*SIG[8]) + (-wa6*SIG[11] - wb6*SIG[10]);

			data[clip[5]+0] = SIG[0] + (wa6*SIG[2] - wb6*SIG[3]) +
							(-wa6*SIG[4] - wb6*SIG[5]) + (-SIG[6])+
							(-wa6*SIG[8] + wb6*SIG[9]) + (wa6*SIG[10] + wb6*SIG[11]); 

			data[clip[5]+1] =  SIG[1] + (wa6*SIG[3] + wb6*SIG[2]) +
							(-wa6*SIG[5] + wb6*SIG[4]) + (-SIG[7])+
							(-wa6*SIG[9] -	wb6*SIG[8]) + (wa6*SIG[11] - wb6*SIG[10]);
		
			break;
		case 7:	
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] 
							+ SIG[8] + SIG[10] + SIG[12];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] 
							+ SIG[9] + SIG[11] + SIG[13];
			data[clip[1]+0] = SIG[0] + (wa7*SIG[2] + wb7*SIG[3]) 
							+(-wc7*SIG[4] + wd7*SIG[5]) 
							+ (-we7*SIG[6] + wf7*SIG[7]) 
							+ (-we7*SIG[8] - wf7*SIG[9]) 
							+ (-wc7*SIG[10] - wd7*SIG[11]) 
							+ (wa7*SIG[12] - wb7*SIG[13]);
			data[clip[1]+1] = SIG[1] + (wa7*SIG[3] - wb7*SIG[2]) 
							+(-wc7*SIG[5] - wd7*SIG[4]) 
							+ (-we7*SIG[7] - wf7*SIG[6]) 
							+ (-we7*SIG[9] + wf7*SIG[8]) 
							+ (-wc7*SIG[11] + wd7*SIG[10]) 
							+ (wa7*SIG[13] + wb7*SIG[12]);
			data[clip[2]+0] = SIG[0] + (-wc7*SIG[2] + wd7*SIG[3]) 
							+(-we7*SIG[4] - wf7*SIG[5]) 
							+ (wa7*SIG[6] - wb7*SIG[7]) 
							+ (wa7*SIG[8] + wb7*SIG[9]) 
							+ (-we7*SIG[10] + wf7*SIG[11]) 
							+ (-wc7*SIG[12] - wd7*SIG[13]);
			data[clip[2]+1] = SIG[1] + (-wc7*SIG[3] - wd7*SIG[2]) 
							+(-we7*SIG[5] + wf7*SIG[4]) 
							+ (wa7*SIG[7] + wb7*SIG[6]) 
							+ (wa7*SIG[9] - wb7*SIG[8]) 
							+ (-we7*SIG[11] - wf7*SIG[10]) 
							+ (-wc7*SIG[13] + wd7*SIG[12]);
			data[clip[3]+0] = SIG[0] + (-we7*SIG[2] + wf7*SIG[3]) 
							+(wa7*SIG[4] - wb7*SIG[5]) 
							+ (-wc7*SIG[6] + wd7*SIG[7]) 
							+ (-wc7*SIG[8] - wd7*SIG[9]) 
							+ (wa7*SIG[10] + wb7*SIG[11]) 
							+ (-we7*SIG[12] - wf7*SIG[13]);
			data[clip[3]+1] = SIG[1] + (-we7*SIG[3] - wf7*SIG[2]) 
							+(wa7*SIG[5] + wb7*SIG[4]) 
							+ (-wc7*SIG[7] - wd7*SIG[6]) 
							+ (-wc7*SIG[9] + wd7*SIG[8]) 
							+ (wa7*SIG[11] - wb7*SIG[10]) 
							+ (-we7*SIG[13] + wf7*SIG[12]);
			data[clip[4]+0] = SIG[0] + (-we7*SIG[2] - wf7*SIG[3]) 
							+(wa7*SIG[4] + wb7*SIG[5]) 
							+ (-wc7*SIG[6] - wd7*SIG[7]) 
							+ (-wc7*SIG[8] + wd7*SIG[9]) 
							+ (wa7*SIG[10] - wb7*SIG[11]) 
							+ (-we7*SIG[12] + wf7*SIG[13]);
			data[clip[4]+1] = SIG[1] + (-we7*SIG[3] + wf7*SIG[2]) 
							+(wa7*SIG[5] - wb7*SIG[4]) 
							+ (-wc7*SIG[7] + wd7*SIG[6]) 
							+ (-wc7*SIG[9] - wd7*SIG[8]) 
							+ (wa7*SIG[11] + wb7*SIG[10]) 
							+ (-we7*SIG[13] - wf7*SIG[12]);
			data[clip[5]+0] = SIG[0] + (-wc7*SIG[2] - wd7*SIG[3]) 
							+(-we7*SIG[4] + wf7*SIG[5]) 
							+ (wa7*SIG[6] + wb7*SIG[7]) 
							+ (wa7*SIG[8] - wb7*SIG[9]) 
							+ (-we7*SIG[10] - wf7*SIG[11]) 
							+ (-wc7*SIG[12] + wd7*SIG[13]);
			data[clip[5]+1] = SIG[1] + (-wc7*SIG[3] + wd7*SIG[2]) 
							+(-we7*SIG[5] - wf7*SIG[4]) 
							+ (wa7*SIG[7] - wb7*SIG[6]) 
							+ (wa7*SIG[9] + wb7*SIG[8]) 
							+ (-we7*SIG[11] + wf7*SIG[10]) 
							+ (-wc7*SIG[13] - wd7*SIG[12]);
			data[clip[6]+0] = SIG[0] + (wa7*SIG[2] - wb7*SIG[3]) 
							+(-wc7*SIG[4] - wd7*SIG[5]) 
							+ (-we7*SIG[6] - wf7*SIG[7]) 
							+ (-we7*SIG[8] + wf7*SIG[9]) 
							+ (-wc7*SIG[10] + wd7*SIG[11]) 
							+ (wa7*SIG[12] + wb7*SIG[13]);
			data[clip[6]+1] = SIG[1] + (wa7*SIG[3] + wb7*SIG[2]) 
							+(-wc7*SIG[5] + wd7*SIG[4]) 
							+ (-we7*SIG[7] + wf7*SIG[6]) 
							+ (-we7*SIG[9] - wf7*SIG[8]) 
							+ (-wc7*SIG[11] - wd7*SIG[10]) 
							+ (wa7*SIG[13] - wb7*SIG[12]);
			break;
		case 10:
			data[clip[0]+0] = SIG[0] + SIG[2] + SIG[4] + SIG[6] + SIG[8]
							+ SIG[10] + SIG[12] + SIG[14] + SIG[16] + SIG[18];
			data[clip[0]+1] = SIG[1] + SIG[3] + SIG[5] + SIG[7] + SIG[9]
							+ SIG[11] + SIG[13] + SIG[15] + SIG[17] + SIG[19];
	
			data[clip[1]+0] = SIG[0] + (wa10*SIG[2] + wb10*SIG[3])
							+ (wc10*SIG[4] + wd10*SIG[5]) + (-wc10*SIG[6]
							+ wd10*SIG[7]) + (-wa10*SIG[8] + wb10*SIG[9])
							+ (-SIG[10]) + (-wa10*SIG[12] - wb10*SIG[13])
							+ (-wc10*SIG[14] - wd10*SIG[15]) + (wc10*SIG[16]
							- wd10*SIG[17]) + (wa10*SIG[18] - wb10*SIG[19]);
							
			data[clip[1]+1]  = SIG[1] + (wa10*SIG[3] - wb10*SIG[2])
							+(wc10*SIG[5] - wd10*SIG[4]) + (-wc10*SIG[7]
							- wd10*SIG[6]) + (-wa10*SIG[9] - wb10*SIG[8])
							+ (-SIG[11]) + (-wa10*SIG[13] + wb10*SIG[12])
							+ (-wc10*SIG[15] + wd10*SIG[14]) + (wc10*SIG[17]
							+ wd10*SIG[16]) +(wa10*SIG[19] + wb10*SIG[18]) ;

			data[clip[2]+0] = SIG[0] + (wc10*SIG[2] + wd10*SIG[3])
							+(-wa10*SIG[4] + wb10*SIG[5]) + (-wa10*SIG[6]
							- wb10*SIG[7]) + (wc10*SIG[8] - wd10*SIG[9])
							+ (SIG[10]) + (wc10*SIG[12] + 	wd10*SIG[13])
							+ (-wa10*SIG[14] + wb10*SIG[15]) + (-wa10*SIG[16]
							- wb10*SIG[17]) + (wc10*SIG[18] - wd10*SIG[19]);
			data[clip[2]+1] = SIG[1] + (wc10*SIG[3] - wd10*SIG[2])
							+(-wa10*SIG[5] - wb10*SIG[4]) + (-wa10*SIG[7]
							+ wb10*SIG[6]) + (wc10*SIG[9] + wd10*SIG[8])
							+ (SIG[11]) + (wc10*SIG[13] - wd10*SIG[12])
							+ (-wa10*SIG[15] - wb10*SIG[14]) + (-wa10*SIG[17]
							+ wb10*SIG[16]) +(wc10*SIG[19] + wd10*SIG[18]) ;


			data[clip[3]+0] = SIG[0] + (-wc10*SIG[2] + wd10*SIG[3]) 
							+(-wa10*SIG[4] - wb10*SIG[5]) + (wa10*SIG[6]
							- wb10*SIG[7]) + (wc10*SIG[8] + wd10*SIG[9]) 
							+ (-SIG[10]) + (wc10*SIG[12] - 	wd10*SIG[13]) 
							+ (wa10*SIG[14] + wb10*SIG[15]) + (-wa10*SIG[16]
							+ wb10*SIG[17]) + (-wc10*SIG[18] - wd10*SIG[19]);
			data[clip[3]+1] = SIG[1] + (-wc10*SIG[3] - wd10*SIG[2]) 
							+(-wa10*SIG[5] + wb10*SIG[4]) + (wa10*SIG[7] 
							+ wb10*SIG[6]) + (wc10*SIG[9] - wd10*SIG[8])
							+ (-SIG[11]) + (wc10*SIG[13] + wd10*SIG[12]) 
							+ (wa10*SIG[15] - wb10*SIG[14]) + (-wa10*SIG[17]
							- wb10*SIG[16]) +(-wc10*SIG[19] + wd10*SIG[18]) ;

			data[clip[4]+0] = SIG[0] + (-wa10*SIG[2] + wb10*SIG[3]) 
							+ (wc10*SIG[4] - wd10*SIG[5]) + (wc10*SIG[6]
							+ wd10*SIG[7]) + (-wa10*SIG[8] - wb10*SIG[9]) 
							+ (SIG[10]) + (-wa10*SIG[12] + 	wb10*SIG[13]) 
							+ (wc10*SIG[14] - wd10*SIG[15]) + (wc10*SIG[16]
							+ wd10*SIG[17]) + (-wa10*SIG[18] - wb10*SIG[19]);
			data[clip[4]+1]	= SIG[1] + (-wa10*SIG[3] - wb10*SIG[2]) 
							+ (wc10*SIG[5] + wd10*SIG[4]) + (wc10*SIG[7]
							- wd10*SIG[6]) + (-wa10*SIG[9] + wb10*SIG[8])
							+ (SIG[11]) + (-wa10*SIG[13] - wb10*SIG[12]) 
							+ (wc10*SIG[15] + wd10*SIG[14])	+ (wc10*SIG[17]
							- wd10*SIG[16]) + (-wa10*SIG[19] + wb10*SIG[18]) ;

			data[clip[5]+0] = SIG[0] + (-SIG[2]) + SIG[4] + (-SIG[6])
							+ SIG[8] + (-SIG[10]) + SIG[12] + (-SIG[14])
							+ SIG[16] + (-SIG[18]);
			data[clip[5]+1] = SIG[1] + (-SIG[3]) + SIG[5] + (-SIG[7])
							+ SIG[9] + (-SIG[11]) + SIG[13] + (-SIG[15])
							+ SIG[17] + (-SIG[19]);
	
			data[clip[6]+0] = SIG[0] + (-wa10*SIG[2] - wb10*SIG[3])
							+(wc10*SIG[4] + wd10*SIG[5]) + (wc10*SIG[6]
							- wd10*SIG[7]) + (-wa10*SIG[8] + wb10*SIG[9])
							+ (SIG[10]) + (-wa10*SIG[12] - wb10*SIG[13])
							+ (wc10*SIG[14] + wd10*SIG[15]) + (wc10*SIG[16]
							- wd10*SIG[17]) + (-wa10*SIG[18] + wb10*SIG[19]);
			data[clip[6]+1] = SIG[1] + (-wa10*SIG[3] + wb10*SIG[2])
							+(wc10*SIG[5] - wd10*SIG[4]) + (wc10*SIG[7]
							+ wd10*SIG[6]) + (-wa10*SIG[9] - wb10*SIG[8])
							+ (SIG[11]) + (-wa10*SIG[13] + wb10*SIG[12])
							+ (wc10*SIG[15] - wd10*SIG[14]) + (wc10*SIG[17]
							+ wd10*SIG[16]) +(-wa10*SIG[19] - wb10*SIG[18]) ;


			data[clip[7]+0] = SIG[0] + (-wc10*SIG[2] - wd10*SIG[3])
							+(-wa10*SIG[4] + wb10*SIG[5]) + (wa10*SIG[6]
							+ wb10*SIG[7]) + (wc10*SIG[8] - wd10*SIG[9])
							+ (-SIG[10]) + (wc10*SIG[12] + wd10*SIG[13])
							+ (wa10*SIG[14] - wb10*SIG[15]) + (-wa10*SIG[16]
							- wb10*SIG[17]) + (-wc10*SIG[18] + wd10*SIG[19]);
			data[clip[7]+1] = SIG[1] + (-wc10*SIG[3] + wd10*SIG[2])
							+(-wa10*SIG[5] -wb10*SIG[4]) + (wa10*SIG[7]
							- wb10*SIG[6]) + (wc10*SIG[9] + wd10*SIG[8])
							+ (-SIG[11]) + (wc10*SIG[13] - wd10*SIG[12])
							+ (wa10*SIG[15] + wb10*SIG[14]) + (-wa10*SIG[17]
							+ wb10*SIG[16]) +(-wc10*SIG[19] - wd10*SIG[18]) ;


			data[clip[8]+0] = SIG[0] + (wc10*SIG[2] - wd10*SIG[3])
							+(-wa10*SIG[4] - wb10*SIG[5]) + (-wa10*SIG[6]
							+ wb10*SIG[7]) + (wc10*SIG[8] + wd10*SIG[9])
							+ (SIG[10]) + (wc10*SIG[12] - wd10*SIG[13])
							+ (-wa10*SIG[14] - wb10*SIG[15]) + (-wa10*SIG[16]
							+ wb10*SIG[17]) + (wc10*SIG[18] + wd10*SIG[19]);
			data[clip[8]+1] = SIG[1] + (wc10*SIG[3] + wd10*SIG[2])
							+(-wa10*SIG[5] + wb10*SIG[4]) + (-wa10*SIG[7]
							- wb10*SIG[6]) + (wc10*SIG[9] - wd10*SIG[8])
							+ (SIG[11]) + (wc10*SIG[13] + wd10*SIG[12])
							+ (-wa10*SIG[15] + wb10*SIG[14]) + (-wa10*SIG[17]
							- wb10*SIG[16]) +(wc10*SIG[19] - wd10*SIG[18]) ;


			data[clip[9]+0] = SIG[0] + (wa10*SIG[2] - wb10*SIG[3])
							+(wc10*SIG[4] - wd10*SIG[5]) + (-wc10*SIG[6]
							- wd10*SIG[7]) + (-wa10*SIG[8] - wb10*SIG[9])
							+ (-SIG[10]) + (-wa10*SIG[12] + wb10*SIG[13])
							+ (-wc10*SIG[14] + wd10*SIG[15]) + (wc10*SIG[16]
							+ wd10*SIG[17]) + (wa10*SIG[18] + wb10*SIG[19]);
			data[clip[9]+1] = SIG[1] + (wa10*SIG[3] + wb10*SIG[2])
							+(wc10*SIG[5] + wd10*SIG[4]) + (-wc10*SIG[7]
							+ wd10*SIG[6]) + (-wa10*SIG[9] + wb10*SIG[8])
							+ (-SIG[11]) + (-wa10*SIG[13] - wb10*SIG[12])
							+ (-wc10*SIG[15] - wd10*SIG[14]) + (wc10*SIG[17]
							- wd10*SIG[16]) +(wa10*SIG[19] - wb10*SIG[18]) ;
			break;
	}
	

	


}
__kernel void DIT6C2C(
			__global double *data, 
			const int size,
			unsigned int stage,
			unsigned int dir)
{
	
	int idX = get_global_id(0);

	//int powX = topePow(5.,stage,11);
	int powX = topePowInt(6,stage);
	int powXm1 = powX/6;
	// x =pow(5.0,2)=25 whereas x=pow(5.0f,2)=24
	
	int clipOne, clipTwo, clipThr, clipFou, clipFiv, clipSix;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);
	clipFiv		= 2 * (kIndex + yIndex * powX + 4 * powXm1);
	clipSix		= 2 * (kIndex + yIndex * powX + 5 * powXm1);
	
	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
	double2 SIG6A = (double2)(	data[clipSix+0],data[clipSix+1]);

	double2 clSet2, clSet3, clSet4, clSet5, clSet6, temp2;
	
	if (kIndex!=0) {
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;
		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;
		
		clSet6.x = cos(5*CLPT*kIndex/powX);
		clSet6.y = -sin(5*CLPT*kIndex/powX);
		if (dir == 0) clSet6.y *= -1;
		TEMPC.x = SIG6A.x * clSet6.x - SIG6A.y * clSet6.y;
		TEMPC.y = SIG6A.x * clSet6.y + SIG6A.y * clSet6.x;
		SIG6A.x = TEMPC.x;
		SIG6A.y = TEMPC.y;
	
	}	

	data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x + SIG6A.x;
	data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y + SIG6A.y;

					
	data[clipTwo+0] = SIG4A.s0 + (wa6*SIG4A.s2 + wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 + wb6*SIG4A.s5) + (-SIG4A.s6)+
					(-wa6*SIG5A.x - wb6*SIG5A.y) + (wa6*SIG6A.x - wb6*SIG6A.y); 
	data[clipTwo+1] = SIG4A.s1 + (wa6*SIG4A.s3 - wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 - wb6*SIG4A.s4) + (-SIG4A.s7)+
					(-wa6*SIG5A.y +	wb6*SIG5A.x) + (wa6*SIG6A.y + wb6*SIG6A.x);


					
	data[clipThr+0] = SIG4A.s0 + (-wa6*SIG4A.s2 + wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 - wb6*SIG4A.s5) + (SIG4A.s6)+
					(-wa6*SIG5A.x + wb6*SIG5A.y) + (-wa6*SIG6A.x - wb6*SIG6A.y); 
	data[clipThr+1] = SIG4A.s1 + (-wa6*SIG4A.s3 - wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 + wb6*SIG4A.s4) + (SIG4A.s7)+
					(-wa6*SIG5A.y -	wb6*SIG5A.x) + (-wa6*SIG6A.y + wb6*SIG6A.x);

	data[clipFou+0] = SIG4A.s0 + (-SIG4A.s2) + (SIG4A.s4) + (-SIG4A.s6) +
					(SIG5A.x) + (-SIG6A.x);
	data[clipFou+1] = SIG4A.s1 + (-SIG4A.s3) + (SIG4A.s5) + (-SIG4A.s7) +
					(SIG5A.y) + (-SIG6A.y);

	data[clipFiv+0] = SIG4A.s0 + (-wa6*SIG4A.s2 - wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 + wb6*SIG4A.s5) + (SIG4A.s6)+
					(-wa6*SIG5A.x - wb6*SIG5A.y) + (-wa6*SIG6A.x + wb6*SIG6A.y); 

	data[clipFiv+1] = SIG4A.s1 + (-wa6*SIG4A.s3 + wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 - wb6*SIG4A.s4) + (SIG4A.s7)+
					(-wa6*SIG5A.y +	wb6*SIG5A.x) + (-wa6*SIG6A.y - wb6*SIG6A.x);

	data[clipSix+0] = SIG4A.s0 + (wa6*SIG4A.s2 - wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 - wb6*SIG4A.s5) + (-SIG4A.s6)+
					(-wa6*SIG5A.x + wb6*SIG5A.y) + (wa6*SIG6A.x + wb6*SIG6A.y); 

	data[clipSix+1] =  SIG4A.s1 + (wa6*SIG4A.s3 + wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 + wb6*SIG4A.s4) + (-SIG4A.s7)+
					(-wa6*SIG5A.y -	wb6*SIG5A.x) + (wa6*SIG6A.y - wb6*SIG6A.x);


	#if 0 // Debug code
	data[clipOne+0] = 11;//kIndex;
	data[clipOne+1] = 111;//yIndex;
	data[clipTwo+0] = 22;//kIndex;
	data[clipTwo+1] = 222;//yIndex;
	data[clipThr+0] = 33;//kIndex;
	data[clipThr+1] = 333;//yIndex;
	data[clipFou+0] = 44;//kIndex;
	data[clipFou+1] = 444;//yIndex;
	data[clipFiv+0] = 55;//kIndex;
	data[clipFiv+1] = 555;//yIndex;
	data[clipSix+0] = 66;
	data[clipSix+1] = 666;
	#endif

}

__kernel void DIT5C2C(
			__global double *data, 
			const int size,
			unsigned int stage,
			unsigned int dir)
{
#if 1 // correct
	int idX = get_global_id(0);

	//int powX = topePow(5.,stage,11);
	int powX = topePowInt(5,stage);
	int powXm1 = powX/5;
	// x =pow(5.0,2)=25 whereas x=pow(5.0f,2)=24
	
	int clipOne, clipTwo, clipThr, clipFou, clipFiv;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);
	clipFou		= 2 * (kIndex + yIndex * powX + 3 * powXm1);
	clipFiv		= 2 * (kIndex + yIndex * powX + 4 * powXm1);
	
	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);

	double2 clSet2, clSet3, clSet4, clSet5,temp2;
	
	if (kIndex!=0) {
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;
		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;
	}	

	data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
	data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
	data[clipTwo+0] = SIG4A.s0 + (wa5*SIG4A.s2 + wb5*SIG4A.s3) 
					+(-wc5*SIG4A.s4 + wd5*SIG4A.s5) + (-wc5*SIG4A.s6 
					- wd5*SIG4A.s7)+(wa5*SIG5A.x - 	wb5*SIG5A.y);
	data[clipTwo+1] = SIG4A.s1 + (wa5*SIG4A.s3 - wb5*SIG4A.s2)
					+(-wc5*SIG4A.s5 - wd5*SIG4A.s4) + (-wc5*SIG4A.s7 +
					wd5*SIG4A.s6)+(wa5*SIG5A.y + wb5*SIG5A.x);
	data[clipThr+0] = SIG4A.s0 + (-wc5*SIG4A.s2 + wd5*SIG4A.s3)
					+(wa5*SIG4A.s4 - wb5*SIG4A.s5) + (wa5*SIG4A.s6 +
					wb5*SIG4A.s7)+(-wc5*SIG5A.x - wd5*SIG5A.y); 
	data[clipThr+1] = SIG4A.s1 + (-wc5*SIG4A.s3 - wd5*SIG4A.s2) 
					+(wa5*SIG4A.s5 + wb5*SIG4A.s4) + (wa5*SIG4A.s7 
					- wb5*SIG4A.s6)+(-wc5*SIG5A.y +	wd5*SIG5A.x);
	data[clipFou+0] = SIG4A.s0 + (-wc5*SIG4A.s2 - wd5*SIG4A.s3) 
					+(wa5*SIG4A.s4 + wb5*SIG4A.s5) + (wa5*SIG4A.s6 
					- wb5*SIG4A.s7)+(-wc5*SIG5A.x + wd5*SIG5A.y);
	data[clipFou+1] = SIG4A.s1 + (-wc5*SIG4A.s3 + wd5*SIG4A.s2) 
					+(wa5*SIG4A.s5 - wb5*SIG4A.s4) + (wa5*SIG4A.s7 
					+ wb5*SIG4A.s6)+(-wc5*SIG5A.y - wd5*SIG5A.x);
	data[clipFiv+0] = SIG4A.s0 + (wa5*SIG4A.s2 - wb5*SIG4A.s3) 
					+(-wc5*SIG4A.s4 - wd5*SIG4A.s5) + (-wc5*SIG4A.s6 
					+ wd5*SIG4A.s7)+(wa5*SIG5A.x + 	wb5*SIG5A.y);
	data[clipFiv+1] = SIG4A.s1 + (wa5*SIG4A.s3 + wb5*SIG4A.s2) 
					+(-wc5*SIG4A.s5 + wd5*SIG4A.s4) + (-wc5*SIG4A.s7 
					- wd5*SIG4A.s6)+(wa5*SIG5A.y - wb5*SIG5A.x);

	#if 0 // Debug code
	data[clipOne+0] = 11;//kIndex;
	data[clipOne+1] = 111;//yIndex;
	data[clipTwo+0] = 22;//kIndex;
	data[clipTwo+1] = 222;//yIndex;
	data[clipThr+0] = 33;//kIndex;
	data[clipThr+1] = 333;//yIndex;
	data[clipFou+0] = 44;//kIndex;
	data[clipFou+1] = 444;//yIndex;
	data[clipFiv+0] = 55;//kIndex;
	data[clipFiv+1] = 555;//yIndex;
	#endif
#endif //end correct

}
__kernel void DIT3C2C(
			__global double *data, 
			const int size,
			unsigned int stage,
			unsigned int dir)
{
	int idX = get_global_id(0);
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;

	float powX = 1.0;  //int is giving problem with pow(3.0,powRemain)
	int powXm1 = 1.0;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(3.0,powMaxLvl);
	}
	powX =1* pow(3.0,powRemain);
	
	powXm1 = powX/3;

	int clipOne, clipTwo, clipThr;
	int yIndex, kIndex;
	yIndex = idX / powXm1;
	kIndex = idX % powXm1;
	
	clipOne 	= 2 * (kIndex + yIndex * powX + 0 * powXm1);
	clipTwo 	= 2 * (kIndex + yIndex * powX + 1 * powXm1);
	clipThr		= 2 * (kIndex + yIndex * powX + 2 * powXm1);

	double2 TEMPC;
	double4 SIG3A = (double4)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1]);
	double2 SIG3B =(double2)(	data[clipThr+0],data[clipThr+1]);

	double2 clSet2, clSet3,temp2;
	
	if (kIndex!=0) {
		
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG3A.s2 * clSet2.x - SIG3A.s3 * clSet2.y;
		TEMPC.y = SIG3A.s2 * clSet2.y + SIG3A.s3 * clSet2.x;
		SIG3A.s2 = TEMPC.x;
		SIG3A.s3 = TEMPC.y;
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG3B.x * clSet3.x - SIG3B.y * clSet3.y;
		TEMPC.y = SIG3B.x * clSet3.y + SIG3B.y * clSet3.x;
		SIG3B.x=TEMPC.x;
		SIG3B.y=TEMPC.y;
		
	}	
	data[clipOne+0] = SIG3A.s0 + SIG3A.s2 + SIG3B.x;
	data[clipOne+1] = SIG3A.s1 + SIG3A.s3 + SIG3B.y;
	
	data[clipTwo+0] = 
		SIG3A.s0 + (-0.5*SIG3A.s2 + 0.866* SIG3A.s3) 
		+(-0.5*SIG3B.x - 0.866*SIG3B.y);
	data[clipTwo+1] = 
		SIG3A.s1 + (-0.866*SIG3A.s2 - 0.5 * SIG3A.s3) 
		+(0.866*SIG3B.x - 0.5*SIG3B.y);
	data[clipThr+0] = 
		SIG3A.s0 + (-0.5*SIG3A.s2 - 0.866* SIG3A.s3) 
		+(-0.5*SIG3B.x + 0.866*SIG3B.y);
	data[clipThr+1] = 
		SIG3A.s1 + (0.866*SIG3A.s2 - 0.5* SIG3A.s3) 
		+(-0.866*SIG3B.x - 0.5*SIG3B.y);

		#if 0 //debug
		data[0]=421;
		#endif
}

__kernel void DIT3C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int powX 	= topePow(3., stage, 6);
	int powXm1 	= powX/3;

	int clipOne, clipTwo, clipThr;
	int yIndex, kIndex;

	int BASE 	= 0;
	int STRIDE 	= 1;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}

	clipOne = 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo = 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));

	double2 TEMPC;
	double4 SIG3A = (double4)(	data[clipOne+0],data[clipOne+1],
								data[clipTwo+0],data[clipTwo+1]);
	double2 SIG3B = (double2)(	data[clipThr+0],data[clipThr+1]);

	double2 clSet2, clSet3, temp2;
	
	if (kIndex!=0) {
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG3A.s2 * clSet2.x - SIG3A.s3 * clSet2.y;
		TEMPC.y = SIG3A.s2 * clSet2.y + SIG3A.s3 * clSet2.x;
		SIG3A.s2 = TEMPC.x;
		SIG3A.s3 = TEMPC.y;
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG3B.x * clSet3.x - SIG3B.y * clSet3.y;
		TEMPC.y = SIG3B.x * clSet3.y + SIG3B.y * clSet3.x;
		SIG3B.x=TEMPC.x;
		SIG3B.y=TEMPC.y;
		
	}	
	data[clipOne+0] = SIG3A.s0 + SIG3A.s2 + SIG3B.x;
	data[clipOne+1] = SIG3A.s1 + SIG3A.s3 + SIG3B.y;
	
	data[clipTwo+0] = 
		SIG3A.s0 + (-0.5*SIG3A.s2 + 0.866* SIG3A.s3) 
		+(-0.5*SIG3B.x - 0.866*SIG3B.y);
	data[clipTwo+1] = 
		SIG3A.s1 + (-0.866*SIG3A.s2 - 0.5 * SIG3A.s3) 
		+(0.866*SIG3B.x - 0.5*SIG3B.y);
	data[clipThr+0] = 
		SIG3A.s0 + (-0.5*SIG3A.s2 - 0.866* SIG3A.s3) 
		+(-0.5*SIG3B.x + 0.866*SIG3B.y);
	data[clipThr+1] = 
		SIG3A.s1 + (0.866*SIG3A.s2 - 0.5* SIG3A.s3) 
		+(-0.866*SIG3B.x - 0.5*SIG3B.y);

		#if 0  //debug
		data[clipOne+0]=11;
		data[clipOne+1]=11;
		data[clipTwo+0]=22;
		data[clipTwo+1]=22;
		data[clipThr+0]=33;
		data[clipThr+1]=33;
		#endif
		#if 0  //debug
		data[2*(idY*x+idX)]    = powX;
		data[2*(idY*x+idX)+1]  = powX;
		#endif

}

__kernel void DIT4C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int powX = topePow(4., stage, 7);
	int powXm1 = powX/4;

	int clipOne, clipTwo, clipThr, clipFou, yIndex, kIndex, coeffUse;
	
	int BASE 	= 0;
	int STRIDE 	= 1;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				coeffUse 	= kIndex * (x / powX);
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				coeffUse 	= kIndex * (y / powX);
				break;
	}

	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));

	double8 SIGA = (double8)(	data[clipOne+0],	data[clipOne+1],
								data[clipTwo+0],	data[clipTwo+1],
								data[clipThr+0],	data[clipThr+1],
								data[clipFou+0],	data[clipFou+1]	);
	
	double2 TEMPC, clSet1;

	double2 clSet2, clSet3;
	if (kIndex != 0) {
		clSet2.x =  cos(2.*CLPT*kIndex/powX);
		clSet2.y = -sin(2.*CLPT*kIndex/powX);
		TEMPC.x = SIGA.s2 * clSet2.x - SIGA.s3 * clSet2.y;
		TEMPC.y = SIGA.s3 * clSet2.x + SIGA.s2 * clSet2.y;
		SIGA.s2 = TEMPC.x;
		SIGA.s3 = TEMPC.y;
		clSet1.x =  cos(CLPT*kIndex/powX);
		clSet1.y = -sin(CLPT*kIndex/powX);
		TEMPC.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TEMPC.y = SIGA.s5 * clSet1.x + SIGA.s4 * clSet1.y;
		SIGA.s4 = TEMPC.x;
		SIGA.s5 = TEMPC.y;
		clSet3.x =  cos(3.*CLPT*kIndex/powX);
		clSet3.y = -sin(3.*CLPT*kIndex/powX);
		TEMPC.x = SIGA.s6 * clSet3.x - SIGA.s7 * clSet3.y;
		TEMPC.y = SIGA.s7 * clSet3.x + SIGA.s6 * clSet3.y;
		SIGA.s6 = TEMPC.x;
		SIGA.s7 = TEMPC.y;
	}	
	
	if (dir == 1) {
		data[clipOne+0] = SIGA.s0 + SIGA.s2 + SIGA.s4 + SIGA.s6;
		data[clipOne+1] = SIGA.s1 + SIGA.s3 + SIGA.s5 + SIGA.s7;
		data[clipTwo+0] = SIGA.s0 - SIGA.s2 + SIGA.s5 - SIGA.s7;
		data[clipTwo+1] = SIGA.s1 - SIGA.s3 - SIGA.s4 + SIGA.s6;
		data[clipThr+0] = SIGA.s0 + SIGA.s2 - SIGA.s4 - SIGA.s6;
		data[clipThr+1] = SIGA.s1 + SIGA.s3 - SIGA.s5 - SIGA.s7;
		data[clipFou+0] = SIGA.s0 - SIGA.s2 - SIGA.s5 + SIGA.s7;
		data[clipFou+1] = SIGA.s1 - SIGA.s3 + SIGA.s4 - SIGA.s6;
	}
	else if (dir == 0) {
		data[clipOne+0] = SIGA.s0 + SIGA.s2 + SIGA.s4 + SIGA.s6;
		data[clipOne+1] = SIGA.s1 + SIGA.s3 + SIGA.s5 + SIGA.s7;
		data[clipTwo+0] = SIGA.s0 - SIGA.s2 - SIGA.s5 + SIGA.s7;
		data[clipTwo+1] = SIGA.s1 - SIGA.s3 + SIGA.s4 - SIGA.s6;
		data[clipThr+0] = SIGA.s0 + SIGA.s2 - SIGA.s4 - SIGA.s6;
		data[clipThr+1] = SIGA.s1 + SIGA.s3 - SIGA.s5 - SIGA.s7;
		data[clipFou+0] = SIGA.s0 - SIGA.s2 + SIGA.s5 - SIGA.s7;
		data[clipFou+1] = SIGA.s1 - SIGA.s3 - SIGA.s4 + SIGA.s6;
	}

	#if 0
	data[clipOne+0] = 11;
	data[clipOne+1] = 0;
	data[clipTwo+0] = 22;
	data[clipTwo+1] = 0;
	data[clipThr+0] = 33;
	data[clipThr+1] = 0;
	data[clipFou+0] = 44;
	data[clipFou+1] = 0;
	#endif
}

__kernel void DIT5C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int powX = topePow(5.,stage,1);
	int powXm1 = powX/5;

	int clipOne, clipTwo, clipThr, clipFou, clipFiv;

	int BASE 	= 0;
	int STRIDE 	= 1;

	int yIndex, kIndex;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}

	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));
	clipFiv		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 4 * powXm1));
	
	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);

	double2 clSet2, clSet3, clSet4, clSet5,temp2;
	
	if (kIndex!=0) {
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;
		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;
	}	

	data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
	data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
	data[clipTwo+0] = SIG4A.s0 + (wa5*SIG4A.s2 + wb5*SIG4A.s3) 
					+(-wc5*SIG4A.s4 + wd5*SIG4A.s5) + (-wc5*SIG4A.s6 
					- wd5*SIG4A.s7)+(wa5*SIG5A.x - 	wb5*SIG5A.y);
	data[clipTwo+1] = SIG4A.s1 + (wa5*SIG4A.s3 - wb5*SIG4A.s2)
					+(-wc5*SIG4A.s5 - wd5*SIG4A.s4) + (-wc5*SIG4A.s7 +
					wd5*SIG4A.s6)+(wa5*SIG5A.y + wb5*SIG5A.x);
	data[clipThr+0] = SIG4A.s0 + (-wc5*SIG4A.s2 + wd5*SIG4A.s3)
					+(wa5*SIG4A.s4 - wb5*SIG4A.s5) + (wa5*SIG4A.s6 +
					wb5*SIG4A.s7)+(-wc5*SIG5A.x - wd5*SIG5A.y); 
	data[clipThr+1] = SIG4A.s1 + (-wc5*SIG4A.s3 - wd5*SIG4A.s2) 
					+(wa5*SIG4A.s5 + wb5*SIG4A.s4) + (wa5*SIG4A.s7 
					- wb5*SIG4A.s6)+(-wc5*SIG5A.y +	wd5*SIG5A.x);
	data[clipFou+0] = SIG4A.s0 + (-wc5*SIG4A.s2 - wd5*SIG4A.s3) 
					+(wa5*SIG4A.s4 + wb5*SIG4A.s5) + (wa5*SIG4A.s6 
					- wb5*SIG4A.s7)+(-wc5*SIG5A.x + wd5*SIG5A.y);
	data[clipFou+1] = SIG4A.s1 + (-wc5*SIG4A.s3 + wd5*SIG4A.s2) 
					+(wa5*SIG4A.s5 - wb5*SIG4A.s4) + (wa5*SIG4A.s7 
					+ wb5*SIG4A.s6)+(-wc5*SIG5A.y - wd5*SIG5A.x);
	data[clipFiv+0] = SIG4A.s0 + (wa5*SIG4A.s2 - wb5*SIG4A.s3) 
					+(-wc5*SIG4A.s4 - wd5*SIG4A.s5) + (-wc5*SIG4A.s6 
					+ wd5*SIG4A.s7)+(wa5*SIG5A.x + 	wb5*SIG5A.y);
	data[clipFiv+1] = SIG4A.s1 + (wa5*SIG4A.s3 + wb5*SIG4A.s2) 
					+(-wc5*SIG4A.s5 + wd5*SIG4A.s4) + (-wc5*SIG4A.s7 
					- wd5*SIG4A.s6)+(wa5*SIG5A.y - wb5*SIG5A.x);

	#if 0 // Debug code
	data[clipOne+0] = powX;//kIndex;
	data[clipOne+1] = 111;//yIndex;
	data[clipTwo+0] = powX;//kIndex;
	data[clipTwo+1] = 222;//yIndex;
	data[clipThr+0] = powX;//kIndex;
	data[clipThr+1] = 333;//yIndex;
	data[clipFou+0] = powX;//kIndex;
	data[clipFou+1] = 444;//yIndex;
	data[clipFiv+0] = powX;//kIndex;
	data[clipFiv+1] = 555;//yIndex;
	#endif
}
__kernel void DIT6C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	//int powX = topePow(5.,stage,11);
	int powX = topePowInt(6,stage);
	int powXm1 = powX/6;
	// x =pow(5.0,2)=25 whereas x=pow(5.0f,2)=24
	
	int clipOne, clipTwo, clipThr, clipFou, clipFiv,clipSix;
	int BASE   = 0;
	int STRIDE = 1;
	int yIndex, kIndex;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}

	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));
	clipFiv		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 4 * powXm1));
	clipSix		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 5 * powXm1));

	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
	double2 SIG6A = (double2)(	data[clipSix+0],data[clipSix+1]);

	double2 clSet2, clSet3, clSet4, clSet5, clSet6, temp2;
	
	if (kIndex!=0) {
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;
		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;
		
		clSet6.x = cos(5*CLPT*kIndex/powX);
		clSet6.y = -sin(5*CLPT*kIndex/powX);
		if (dir == 0) clSet6.y *= -1;
		TEMPC.x = SIG6A.x * clSet6.x - SIG6A.y * clSet6.y;
		TEMPC.y = SIG6A.x * clSet6.y + SIG6A.y * clSet6.x;
		SIG6A.x = TEMPC.x;
		SIG6A.y = TEMPC.y;
	
	}	

	data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x + SIG6A.x;
	data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y + SIG6A.y;

	data[clipTwo+0] = SIG4A.s0 + (wa6*SIG4A.s2 + wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 + wb6*SIG4A.s5) + (-SIG4A.s6)+
					(-wa6*SIG5A.x - wb6*SIG5A.y) + (wa6*SIG6A.x - wb6*SIG6A.y); 
	data[clipTwo+1] = SIG4A.s1 + (wa6*SIG4A.s3 - wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 - wb6*SIG4A.s4) + (-SIG4A.s7)+
					(-wa6*SIG5A.y +	wb6*SIG5A.x) + (wa6*SIG6A.y + wb6*SIG6A.x);


				
	data[clipThr+0] = SIG4A.s0 + (-wa6*SIG4A.s2 + wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 - wb6*SIG4A.s5) + (SIG4A.s6)+
					(-wa6*SIG5A.x + wb6*SIG5A.y) + (-wa6*SIG6A.x - wb6*SIG6A.y); 
	data[clipThr+1] = SIG4A.s1 + (-wa6*SIG4A.s3 - wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 + wb6*SIG4A.s4) + (SIG4A.s7)+
					(-wa6*SIG5A.y -	wb6*SIG5A.x) + (-wa6*SIG6A.y + wb6*SIG6A.x);

	data[clipFou+0] = SIG4A.s0 + (-SIG4A.s2) + (SIG4A.s4) + (-SIG4A.s6) +
					(SIG5A.x) + (-SIG6A.x);
	data[clipFou+1] = SIG4A.s1 + (-SIG4A.s3) + (SIG4A.s5) + (-SIG4A.s7) +
					(SIG5A.y) + (-SIG6A.y);

	data[clipFiv+0] = SIG4A.s0 + (-wa6*SIG4A.s2 - wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 + wb6*SIG4A.s5) + (SIG4A.s6)+
					(-wa6*SIG5A.x - wb6*SIG5A.y) + (-wa6*SIG6A.x + wb6*SIG6A.y); 

	data[clipFiv+1] = SIG4A.s1 + (-wa6*SIG4A.s3 + wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 - wb6*SIG4A.s4) + (SIG4A.s7)+
					(-wa6*SIG5A.y +	wb6*SIG5A.x) + (-wa6*SIG6A.y - wb6*SIG6A.x);

	data[clipSix+0] = SIG4A.s0 + (wa6*SIG4A.s2 - wb6*SIG4A.s3) +
					(-wa6*SIG4A.s4 - wb6*SIG4A.s5) + (-SIG4A.s6)+
					(-wa6*SIG5A.x + wb6*SIG5A.y) + (wa6*SIG6A.x + wb6*SIG6A.y); 

	data[clipSix+1] =  SIG4A.s1 + (wa6*SIG4A.s3 + wb6*SIG4A.s2) +
					(-wa6*SIG4A.s5 + wb6*SIG4A.s4) + (-SIG4A.s7)+
					(-wa6*SIG5A.y -	wb6*SIG5A.x) + (wa6*SIG6A.y - wb6*SIG6A.x);


	#if 0 // Debug code
	data[clipOne+0] = 11;//kIndex;
	data[clipOne+1] = 111;//yIndex;
	data[clipTwo+0] = 22;//kIndex;
	data[clipTwo+1] = 222;//yIndex;
	data[clipThr+0] = 33;//kIndex;
	data[clipThr+1] = 333;//yIndex;
	data[clipFou+0] = 44;//kIndex;
	data[clipFou+1] = 444;//yIndex;
	data[clipFiv+0] = 55;//kIndex;
	data[clipFiv+1] = 555;//yIndex;
	data[clipSix+0] = 66;
	data[clipSix+1] = 666;
	#endif
}
__kernel void DIT7C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

//	int powX = topePow(7.,stage,11);
	int powX = topePowInt(7,stage);	
	int powXm1 = powX/7;
	
	int clipOne, clipTwo, clipThr, clipFou, clipFiv, clipSix, clipSev;
	int yIndex, kIndex;

	int BASE 	= 0;
	int STRIDE 	= 1;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}

	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));
	clipFiv		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 4 * powXm1));
	clipSix		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 5 * powXm1));
	clipSev		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 6 * powXm1));
	
	double2 TEMPC;
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
	double2 SIG6A = (double2)(	data[clipSix+0],data[clipSix+1]);
	double2 SIG7A = (double2)(	data[clipSev+0],data[clipSev+1]);

	double2 clSet2, clSet3, clSet4, clSet5, clSet6, clSet7, temp2;
	
	if (kIndex!=0) {
		
		clSet2.x =  cos(CLPT*kIndex/powX);
		clSet2.y = -sin(CLPT*kIndex/powX);
		if (dir == 0) clSet2.y *= -1;
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2 = TEMPC.x;
		SIG4A.s3 = TEMPC.y;
	
		clSet3.x = cos(2*CLPT*kIndex/powX);
		clSet3.y = -sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet3.y *= -1;
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4 = TEMPC.x;
		SIG4A.s5 = TEMPC.y;

		clSet4.x = cos(3*CLPT*kIndex/powX);
		clSet4.y = -sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet4.y *= -1;
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6 = TEMPC.x;
		SIG4A.s7 = TEMPC.y;
	
		clSet5.x = cos(4*CLPT*kIndex/powX);
		clSet5.y = -sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet5.y *= -1;
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x = TEMPC.x;
		SIG5A.y = TEMPC.y;

		clSet6.x = cos(5*CLPT*kIndex/powX);
		clSet6.y = -sin(5*CLPT*kIndex/powX);
		if (dir == 0) clSet6.y *= -1;
		TEMPC.x = SIG6A.x * clSet6.x - SIG6A.y * clSet6.y;
		TEMPC.y = SIG6A.x * clSet6.y + SIG6A.y * clSet6.x;
		SIG6A.x = TEMPC.x;
		SIG6A.y = TEMPC.y;

		clSet7.x = cos(6*CLPT*kIndex/powX);
		clSet7.y = -sin(6*CLPT*kIndex/powX);
		if (dir == 0) clSet7.y *= -1;
		TEMPC.x = SIG7A.x * clSet7.x - SIG7A.y * clSet7.y;
		TEMPC.y = SIG7A.x * clSet7.y + SIG7A.y * clSet7.x;
		SIG7A.x = TEMPC.x;
		SIG7A.y = TEMPC.y;
	}	
	
	data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 
					+ SIG5A.x + SIG6A.x + SIG7A.x;
	data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 
					+ SIG5A.y + SIG6A.y + SIG7A.y;
	data[clipTwo+0] = SIG4A.s0 + (wa7*SIG4A.s2 + wb7*SIG4A.s3) 
					+(-wc7*SIG4A.s4 + wd7*SIG4A.s5) 
					+ (-we7*SIG4A.s6 + wf7*SIG4A.s7) 
					+ (-we7*SIG5A.x - wf7*SIG5A.y) 
					+ (-wc7*SIG6A.x - wd7*SIG6A.y) 
					+ (wa7*SIG7A.x - wb7*SIG7A.y);
	data[clipTwo+1] = SIG4A.s1 + (wa7*SIG4A.s3 - wb7*SIG4A.s2) 
					+(-wc7*SIG4A.s5 - wd7*SIG4A.s4) 
					+ (-we7*SIG4A.s7 - wf7*SIG4A.s6) 
					+ (-we7*SIG5A.y + wf7*SIG5A.x) 
					+ (-wc7*SIG6A.y + wd7*SIG6A.x) 
					+ (wa7*SIG7A.y + wb7*SIG7A.x);
	data[clipThr+0] = SIG4A.s0 + (-wc7*SIG4A.s2 + wd7*SIG4A.s3) 
					+(-we7*SIG4A.s4 - wf7*SIG4A.s5) 
					+ (wa7*SIG4A.s6 - wb7*SIG4A.s7) 
					+ (wa7*SIG5A.x + wb7*SIG5A.y) 
					+ (-we7*SIG6A.x + wf7*SIG6A.y) 
					+ (-wc7*SIG7A.x - wd7*SIG7A.y);
	data[clipThr+1] = SIG4A.s1 + (-wc7*SIG4A.s3 - wd7*SIG4A.s2) 
					+(-we7*SIG4A.s5 + wf7*SIG4A.s4) 
					+ (wa7*SIG4A.s7 + wb7*SIG4A.s6) 
					+ (wa7*SIG5A.y - wb7*SIG5A.x) 
					+ (-we7*SIG6A.y - wf7*SIG6A.x) 
					+ (-wc7*SIG7A.y + wd7*SIG7A.x);
	data[clipFou+0] = SIG4A.s0 + (-we7*SIG4A.s2 + wf7*SIG4A.s3) 
					+(wa7*SIG4A.s4 - wb7*SIG4A.s5) 
					+ (-wc7*SIG4A.s6 + wd7*SIG4A.s7) 
					+ (-wc7*SIG5A.x - wd7*SIG5A.y) 
					+ (wa7*SIG6A.x + wb7*SIG6A.y) 
					+ (-we7*SIG7A.x - wf7*SIG7A.y);
	data[clipFou+1] = SIG4A.s1 + (-we7*SIG4A.s3 - wf7*SIG4A.s2) 
					+(wa7*SIG4A.s5 + wb7*SIG4A.s4) 
					+ (-wc7*SIG4A.s7 - wd7*SIG4A.s6) 
					+ (-wc7*SIG5A.y + wd7*SIG5A.x) 
					+ (wa7*SIG6A.y - wb7*SIG6A.x) 
					+ (-we7*SIG7A.y + wf7*SIG7A.x);
	data[clipFiv+0] = SIG4A.s0 + (-we7*SIG4A.s2 - wf7*SIG4A.s3) 
					+(wa7*SIG4A.s4 + wb7*SIG4A.s5) 
					+ (-wc7*SIG4A.s6 - wd7*SIG4A.s7) 
					+ (-wc7*SIG5A.x + wd7*SIG5A.y) 
					+ (wa7*SIG6A.x - wb7*SIG6A.y) 
					+ (-we7*SIG7A.x + wf7*SIG7A.y);
	data[clipFiv+1] = SIG4A.s1 + (-we7*SIG4A.s3 + wf7*SIG4A.s2) 
					+(wa7*SIG4A.s5 - wb7*SIG4A.s4) 
					+ (-wc7*SIG4A.s7 + wd7*SIG4A.s6) 
					+ (-wc7*SIG5A.y - wd7*SIG5A.x) 
					+ (wa7*SIG6A.y + wb7*SIG6A.x) 
					+ (-we7*SIG7A.y - wf7*SIG7A.x);
	data[clipSix+0] = SIG4A.s0 + (-wc7*SIG4A.s2 - wd7*SIG4A.s3) 
					+(-we7*SIG4A.s4 + wf7*SIG4A.s5) 
					+ (wa7*SIG4A.s6 + wb7*SIG4A.s7) 
					+ (wa7*SIG5A.x - wb7*SIG5A.y) 
					+ (-we7*SIG6A.x - wf7*SIG6A.y) 
					+ (-wc7*SIG7A.x + wd7*SIG7A.y);
	data[clipSix+1] = SIG4A.s1 + (-wc7*SIG4A.s3 + wd7*SIG4A.s2) 
					+(-we7*SIG4A.s5 - wf7*SIG4A.s4) 
					+ (wa7*SIG4A.s7 - wb7*SIG4A.s6) 
					+ (wa7*SIG5A.y + wb7*SIG5A.x) 
					+ (-we7*SIG6A.y + wf7*SIG6A.x) 
					+ (-wc7*SIG7A.y - wd7*SIG7A.x);
	data[clipSev+0] = SIG4A.s0 + (wa7*SIG4A.s2 - wb7*SIG4A.s3) 
					+(-wc7*SIG4A.s4 - wd7*SIG4A.s5) 
					+ (-we7*SIG4A.s6 - wf7*SIG4A.s7) 
					+ (-we7*SIG5A.x + wf7*SIG5A.y) 
					+ (-wc7*SIG6A.x + wd7*SIG6A.y) 
					+ (wa7*SIG7A.x + wb7*SIG7A.y);
	data[clipSev+1] = SIG4A.s1 + (wa7*SIG4A.s3 + wb7*SIG4A.s2) 
					+(-wc7*SIG4A.s5 + wd7*SIG4A.s4) 
					+ (-we7*SIG4A.s7 + wf7*SIG4A.s6) 
					+ (-we7*SIG5A.y - wf7*SIG5A.x) 
					+ (-wc7*SIG6A.y - wd7*SIG6A.x) 
					+ (wa7*SIG7A.y - wb7*SIG7A.x);
	#if 0 // Debug code
	data[clipOne+0] = 11;//kIndex;
	data[clipOne+1] = 111;//yIndex;
	data[clipTwo+0] = 22;//kIndex;
	data[clipTwo+1] = 222;//yIndex;
	data[clipThr+0] = 33;//kIndex;
	data[clipThr+1] = 333;//yIndex;
	data[clipFou+0] = 44;//kIndex;
	data[clipFou+1] = 444;//yIndex;
	data[clipFiv+0] = 55;//kIndex;
	data[clipFiv+1] = 555;//yIndex;
	#endif
}

__kernel void DIT8C2CM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int stage,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int powX = topePow(8.,stage,4);
	int powXm1 = powX/8;

	int clipOne, clipTwo, clipThr, clipFou, clipFiv, clipSix, clipSev, clipEig;
	int yIndex, kIndex;

	int BASE 	= 0;
	int STRIDE 	= 1;

	switch(type)
	{
		case 0: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;
				break;
		case 1: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				break;
	}
	
	clipOne 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 0 * powXm1));
	clipTwo 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 1 * powXm1));
	clipThr		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 2 * powXm1));
	clipFou		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 3 * powXm1));
	clipFiv		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 4 * powXm1));
	clipSix		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 5 * powXm1));
	clipSev		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 6 * powXm1));
	clipEig		= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + 7 * powXm1));

	double2 CST;
	double2 TMP;
	double2 clSet1;
	double16 SIGA = (double16)(	data[clipOne+0],data[clipOne+1],	// s0, s1
								data[clipTwo+0],data[clipTwo+1],	// s2, s3
								data[clipThr+0],data[clipThr+1],	// s4, s5
								data[clipFou+0],data[clipFou+1],	// s6, s7
								data[clipFiv+0],data[clipFiv+1],	// s8, s9
								data[clipSix+0],data[clipSix+1],	// sa, sb
								data[clipSev+0],data[clipSev+1],	// sc, sd
								data[clipEig+0],data[clipEig+1]);	// se, sf

	if (kIndex!=0) {
		
		clSet1.x =  cos(CLPT*kIndex/powX);
		clSet1.y =  sin(CLPT*kIndex/powX);
		if (dir == 0) clSet1.y *= -1;
		TMP.x = SIGA.s8 * clSet1.x - SIGA.s9 * clSet1.y;
		TMP.y = SIGA.s8 * clSet1.y + SIGA.s9 * clSet1.x;
		SIGA.s8 = TMP.x;
		SIGA.s9 = TMP.y;

		clSet1.x =  cos(2*CLPT*kIndex/powX);
		clSet1.y =  sin(2*CLPT*kIndex/powX);
		if (dir == 0) clSet1.y *= -1;
		TMP.x = SIGA.s4 * clSet1.x - SIGA.s5 * clSet1.y;
		TMP.y = SIGA.s4 * clSet1.y + SIGA.s5 * clSet1.x;
		SIGA.s4 = TMP.x;
		SIGA.s5 = TMP.y;

		clSet1.x =  cos(3*CLPT*kIndex/powX);
		clSet1.y =  sin(3*CLPT*kIndex/powX);
		if (dir == 0) clSet1.y *= -1;
		TMP.x = SIGA.sc * clSet1.x - SIGA.sd * clSet1.y;
		TMP.y = SIGA.sc * clSet1.y + SIGA.sd * clSet1.x;
		SIGA.sc = TMP.x;
		SIGA.sd = TMP.y;

		clSet1.x =  cos(4*CLPT*kIndex/powX);
		clSet1.y =  sin(4*CLPT*kIndex/powX);
		if (dir == 0) clSet1.y *= -1;
		TMP.x = SIGA.s2 * clSet1.x - SIGA.s3 * clSet1.y;
		TMP.y = SIGA.s2 * clSet1.y + SIGA.s3 * clSet1.x;
		SIGA.s2 = TMP.x;
		SIGA.s3 = TMP.y;

		clSet1.x =  cos(5*CLPT*kIndex/powX);
		clSet1.y =  sin(5*CLPT*kIndex/powX);
		if (dir == 0) clSet1.y *= -1;
		TMP.x = SIGA.sa * clSet1.x - SIGA.sb * clSet1.y;
		TMP.y = SIGA.sa * clSet1.y + SIGA.sb * clSet1.x;
		SIGA.sa = TMP.x;
		SIGA.sb = TMP.y;

		clSet1.x =  cos(6*CLPT*kIndex/powX);
		clSet1.y =  sin(6*CLPT*kIndex/powX);
		if (dir == 0) clSet1.y *= -1;
		TMP.x = SIGA.s6 * clSet1.x - SIGA.s7 * clSet1.y;
		TMP.y = SIGA.s6 * clSet1.y + SIGA.s7 * clSet1.x;
		SIGA.s6 = TMP.x;
		SIGA.s7 = TMP.y;

		clSet1.x =  cos(7*CLPT*kIndex/powX);
		clSet1.y =  sin(7*CLPT*kIndex/powX);
		if (dir == 0) clSet1.y *= -1;
		TMP.x = SIGA.se * clSet1.x - SIGA.sf * clSet1.y;
		TMP.y = SIGA.se * clSet1.y + SIGA.sf * clSet1.x;
		SIGA.se = TMP.x;
		SIGA.sf = TMP.y;
	}	

	double16 SIGB = (double16)(	SIGA.s0 + SIGA.s2,
								SIGA.s1 + SIGA.s3,
								SIGA.s0 - SIGA.s2,
								SIGA.s1 - SIGA.s3,
								SIGA.s4 + SIGA.s6,
								SIGA.s5 + SIGA.s7,
								SIGA.s4 - SIGA.s6,
								SIGA.s5 - SIGA.s7,
								SIGA.s8 + SIGA.sa,
								SIGA.s9 + SIGA.sb,
								SIGA.s8 - SIGA.sa,
								SIGA.s9 - SIGA.sb,
								SIGA.sc + SIGA.se,
								SIGA.sd + SIGA.sf,
								SIGA.sc - SIGA.se,
								SIGA.sd - SIGA.sf);

	if (dir == 1) {
		TMP = (double2)((SIGB.sa + SIGB.sb)*d707, (SIGB.sf - SIGB.se)*d707);
		SIGB.sb = (SIGB.sb - SIGB.sa) * d707;
		SIGB.sf = (SIGB.sf + SIGB.se) * -d707;
		SIGB.sa = TMP.x;	
		SIGB.se = TMP.y;
		TMP.x	= SIGB.s7; SIGB.s7 = -SIGB.s6; SIGB.s6 = TMP.x; 
	}
	else if (dir == 0) {
		TMP = (double2)((SIGB.sa - SIGB.sb)*d707, (SIGB.sf + SIGB.se)*-d707);
		SIGB.sb = (SIGB.sb + SIGB.sa) * d707;
		SIGB.sf = (SIGB.se - SIGB.sf) * d707;
		SIGB.sa = TMP.x;
		SIGB.se = TMP.y;
		TMP.x 	= -SIGB.s7; SIGB.s7 = SIGB.s6; SIGB.s6 = TMP.x;
	}

	SIGA.s0 = SIGB.s0 + SIGB.s4;
	SIGA.s1 = SIGB.s1 + SIGB.s5;
	SIGA.s2 = SIGB.s2 + SIGB.s6;
	SIGA.s3 = SIGB.s3 + SIGB.s7;
	SIGA.s4 = SIGB.s0 - SIGB.s4;
	SIGA.s5 = SIGB.s1 - SIGB.s5;
	SIGA.s6 = SIGB.s2 - SIGB.s6;
	SIGA.s7 = SIGB.s3 - SIGB.s7;
	SIGA.s8 = SIGB.s8 + SIGB.sc;
	SIGA.s9 = SIGB.s9 + SIGB.sd;
	SIGA.sa = SIGB.sa + SIGB.se;
	SIGA.sb = SIGB.sb + SIGB.sf;
	if (dir == 1) {
		SIGA.sc = SIGB.s9 - SIGB.sd;
		SIGA.sd = SIGB.sc - SIGB.s8;
		SIGA.se = SIGB.sb - SIGB.sf;
		SIGA.sf = SIGB.se - SIGB.sa;
	}
	else if (dir == 0) {
		SIGA.sc = SIGB.sd - SIGB.s9;
		SIGA.sd = SIGB.s8 - SIGB.sc;
		SIGA.se = SIGB.sf - SIGB.sb;
		SIGA.sf = SIGB.sa - SIGB.se;
	}

	#if 1
	data[clipOne+0] = SIGA.s0 + SIGA.s8;
	data[clipOne+1] = SIGA.s1 + SIGA.s9;
	data[clipTwo+0] = SIGA.s2 + SIGA.sa;
	data[clipTwo+1] = SIGA.s3 + SIGA.sb;
	data[clipThr+0] = SIGA.s4 + SIGA.sc;
	data[clipThr+1] = SIGA.s5 + SIGA.sd;
	data[clipFou+0] = SIGA.s6 + SIGA.se;
	data[clipFou+1] = SIGA.s7 + SIGA.sf;
	data[clipFiv+0] = SIGA.s0 - SIGA.s8;
	data[clipFiv+1] = SIGA.s1 - SIGA.s9;
	data[clipSix+0] = SIGA.s2 - SIGA.sa;
	data[clipSix+1] = SIGA.s3 - SIGA.sb;
	data[clipSev+0] = SIGA.s4 - SIGA.sc;
	data[clipSev+1] = SIGA.s5 - SIGA.sd;
	data[clipEig+0] = SIGA.s6 - SIGA.se; 
	data[clipEig+1] = SIGA.s7 - SIGA.sf;
	#endif
	#if 0
	data[clipOne+0] = 0;//
	data[clipOne+1] = 0;
	data[clipTwo+0] = 0;
	data[clipTwo+1] = 0;
	data[clipThr+0] = 0;
	data[clipThr+1] = 0;
	data[clipFou+0] = 0;
	data[clipFou+1] = 0;
	data[clipFiv+0] = 0;//
	data[clipFiv+1] = 0;
	data[clipSix+0] = 0;
	data[clipSix+1] = 0;
	data[clipSev+0] = 0;
	data[clipSev+1] = 0;
	data[clipEig+0] = idZ;
	data[clipEig+1] = 0;
	#endif
	#if 0
	data[2*(idZ*x*y+idY*x+idX)] = powX;
	data[2*(idZ*x*y+idY*x+idX)+1] = 0;
	#endif
}

__kernel void kernelMUL( __global double *data,
						const int facX, const int facY)
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int POS = idY*facX+idX;

	double2 TMP = (double2)(data[2*POS],data[2*POS+1]);
	double2 TWD = (double2)(cos(CLPT*idX*idY/(facX*facY)),
							-sin(CLPT*idX*idY/(facX*facY)));

	data[2*POS+0] = TMP.x * TWD.x - TMP.y * TWD.y;
	data[2*POS+1] = TMP.y * TWD.x + TMP.x * TWD.y;
}

__kernel void transpose2( __global double *data,
						  const int xR,
						  const int yR)
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);
	int xxx = idY*xR + idX;

	int var1 = xxx;
	int var2 = 0;
	do {	
		var2++;
		var1 = (var1 % yR) * xR + var1 / yR;
	} while (var1 > xxx);

	double temp1, temp2;
	if (var1 < xxx || var2 == 1) {
	}
	else {
		var1 = xxx;	
		temp1 = data[2*var1];
		temp2 = data[2*var1+1];
		do {
			var2 = (var1 % yR) * xR + var1 / yR;
			data[2*var1]   = (var2 == xxx) ? temp1 : data[2*var2];
			data[2*var1+1] = (var2 == xxx) ? temp2 : data[2*var2+1];
			var1 = var2;
		} while (var1 > xxx);
	}
}

__kernel void swapkernel(	__global double *data,	// initial data
						const int x,	// dims
						const int y,
						__global int *bit,	// bitrev data
						const unsigned int type) // x or y or z
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	__private int BASE = 0;
	__private int STRIDE = 1;
	__private double holder;
	__private int runner = 0;
	__private int OLD = 0, NEW = 0;

	switch(type)
	{
		case 0: BASE = idY*x;
				if (idX < bit[idX]) {
					OLD = 2*(BASE+STRIDE*idX);
					NEW = 2*(BASE+STRIDE*bit[idX]);
					
					holder = data[NEW];
					data[NEW] = data[OLD];
					data[OLD] = holder;

					holder = data[NEW+1];
					data[NEW+1] = data[OLD+1];
					data[OLD+1] = holder;
				}
				break;
		case 1: BASE = idX; STRIDE = x; 
				if (idY < bit[idY]) {
					OLD = 2*(BASE+STRIDE*idY);
					NEW = 2*(BASE+STRIDE*bit[idY]);

					holder = data[NEW];
					data[NEW] = data[OLD];
					data[OLD] = holder;

					holder = data[NEW+1];
					data[NEW+1] = data[OLD+1];
					data[OLD+1] = holder;
				}
				break;
	}
}

__kernel void DIT2C2C(	__global double *data, 
						__global double2 *twiddle,
						const int size, unsigned int stage,
						unsigned int dir,
						unsigned int useTwiddle) 
{
	int idX = get_global_id(0);

	int powX = topePow(2., stage, 11);
	int powXm1 = powX/2;

	int yIndex = idX / powXm1;
	int kIndex = idX % powXm1;

	int clipStart 	= 2*(kIndex + yIndex * powX);
	int clipEnd 	= 2*(kIndex + yIndex * powX + powXm1);
	int coeffUse 	= kIndex * (size/powX);

	int red = size/4;
	double2 clSet1;

	int quad = coeffUse/red;
	int buad = coeffUse%red;

	if (useTwiddle == 1) {
		switch(quad) {
			case 0:	clSet1 = (double2)( twiddle[buad].x,  twiddle[buad].y); break;
			case 1: clSet1 = (double2)( twiddle[buad].y, -twiddle[buad].x); break;
			case 2:	clSet1 = (double2)(-twiddle[buad].x, -twiddle[buad].y); break;
			case 3:	clSet1 = (double2)(-twiddle[buad].y,  twiddle[buad].x); break;
		}
	}
	else {
		clSet1.x 	= cos(CLPT*kIndex/powX);
		clSet1.y 	= sin(CLPT*kIndex/powX);
	}
	if (dir == 0) clSet1.y *= -1;

	double4 LOC = (double4)(	data[clipStart + 0],	data[clipStart + 1],
								data[clipEnd + 0],		data[clipEnd + 1]);

	double4 FIN = rad2(LOC,clSet1);

	data[clipStart + 0] = FIN.x;
	data[clipStart + 1] = FIN.y;
	data[clipEnd + 0] 	= FIN.z;
	data[clipEnd + 1] 	= FIN.w;
}

__kernel void divide1D(	__global double2 *data, const int size)
{
	int idX = get_global_id(0);
	data[idX] /= size;
}

__kernel void swap1D(	__global double *data, 
						__global int *bitRev) 
{
	int idX = get_global_id(0);
	double holder;
	int old = 0, new = 0;

	if (idX < bitRev[idX]) {
		old = 2*idX;
		new = 2*bitRev[idX];

		holder = data[new];
		data[new] = data[old];
		data[old] = holder;

		holder = data[new+1];
		data[new+1] = data[old+1];
		data[old+1] = holder;
	}
}

__kernel void reverse2(__global int *bitRev, int logSize)
{
	int global_id = get_global_id(0);

	int powMaxLvl = 11;
	int powLevels, powRemain, powX, x;

	int i, j, andmask, sum = 0, k;
	for (i = logSize - 1, j = 0; i >= 0; i--, j++) {
		andmask = 1 << i;
		k = global_id & andmask;
		powLevels = j / powMaxLvl;
		powRemain = j % powMaxLvl;
		powX = 1;
		for (x = 0; x < powLevels; x++) 
			powX *= pow(2.0f,powMaxLvl);
		powX *= pow(2.0f,powRemain);
		sum += k == 0 ? 0 : powX;
	}
	bitRev[global_id] = sum;
	bitRev[global_id+get_global_size(0)] = sum+1;
}

__kernel void reversen( __global int *bitRev, 
						__local int *bitArray, 
						int logSize, 
						int radix)
{	
	
	//if(radix !=3)
	{	
	#if 1 //[WORKING FOR ALL]
	int idX = get_global_id(0);
	int locX = get_local_id(0);

	int n = idX;

	int j, i;
	for (j = locX*logSize; j < locX*logSize+logSize; j++)
	{
		bitArray[j] = n % radix;
		n = n/radix;
	}	
	int tempRev = 0;
	for( j = locX*logSize+logSize-1, i = 0; j >= locX*logSize; j--, i++)
	{
		//tempRev += bitArray[j] * topePow(radix,i,5);
		tempRev += bitArray[j] * topePowInt(radix,i);
	}
	bitRev[idX] = tempRev;
	//data[1]=420;
	#endif
	}
	
	#if 0 /* not working for radix-3, working for radix-5] 
		   * This is working (but check for extremely large sizes)
		   */
	int idX = get_global_id(0);
	int locX = get_local_id(0);

	int n = idX;

	int j, i;
	for (j = locX*logSize; j < locX*logSize+logSize; j++)
	{
		bitArray[j] = n % radix;
		n = n/radix;
	}
	
	int tempRev = 0;
	for( j = locX*logSize+logSize-1, i = 0; j >= locX*logSize; j--, i++)
	{
		tempRev += bitArray[j] * (int)pow((double)radix,i);
	}
	bitRev[idX] = tempRev;
	#endif
	//else{
	#if 0 /* [working for radix-3, not working for radix-5]
		   * This is also working (but good for large sizes)
		   */
	int idX = get_global_id(0);
	int idL = get_local_id(0);

	int base = logSize*idL;
	int highEnd = base+logSize-1;
	int lowEnd = base;
	int subst = 0;
	int size = idX;

	if (size < radix) {
		bitArray[lowEnd] = size;
	}
	else {
		while (size > radix-1) {
			bitArray[lowEnd] = size % radix;
			lowEnd++;
			size /= radix;
		}
		bitArray[lowEnd] = size;
	}
	while (highEnd != lowEnd) {
		lowEnd++;
		bitArray[lowEnd] = 0;
	}
	int x,y;
	for (x = 0,y=logSize-1; x < logSize; x++,y--) {
		#if 1 // this will help in dealing with large sizes
		int powMaxLvl = 11; 
		int powLevels = x / powMaxLvl;
		int powRemain = x % powMaxLvl;
		int powX = 1;
		int x;
		for (x = 0; x < powLevels; x++)	
			powX *= pow((float)radix,powMaxLvl);
		powX *= pow((float)radix,powRemain);
		subst += bitArray[base+y] * powX;
		#endif
		#if 0 // This will also help in dealing with large sizes
		subst += bitArray[base+y] * (int)exp2(log2((float)radix)*x);
		#endif
	}
	bitRev[idX] = subst;
	#endif
	//}
}

__kernel void scratchToData( __global double2 *data,
							 __global double2 *scratch )
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int index = idY*get_global_size(0) + idX;
	data[index] = scratch[index];
}

__kernel void DFTM(	__global double *data,
						const int x, const int y,
						unsigned int dir,
						unsigned int type,
					__global double *scratch)
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int sizeX = get_global_size(0);
	int sizeY = get_global_size(1);

	int i, id, size;
	int BASE = 0;
	int STRIDE = 1;
	int index1, index2;
	double2 W, TEMP = (double2)(0,0);

	switch(type) 
	{
		case 0: BASE = idY*x; 				size = sizeX; id = idX; break;
		case 1: BASE = idX;		STRIDE = x; size = sizeY; id = idY; break;
	}

	for (i = 0; i < size; i++) 
	{
		W = (double2)( 	cos(CLPT*id*i/size), -sin(CLPT*id*i/size));
		index1 = 2*(BASE+STRIDE*i);
		index2 = 2*(idY*x+idX);
		TEMP.x += data[index1] * W.x - data[index1+1] * W.y;
		TEMP.y += data[index1] * W.y + data[index1+1] * W.x;
	}
	//scratch[2*(BASE+STRIDE*idX)]   = TEMP.x;
	//scratch[2*(BASE+STRIDE*idX)+1] = TEMP.y;
	scratch[index2]   = TEMP.x;
	scratch[index2+1] = TEMP.y;

	#if 0 // Debug
	scratch[2*(idY*x+idX)] 	 = 11;//scratch[2*idX];
	scratch[2*(idY*x+idX)+1]   = 22;//scratch[2*idX+1];
	#endif
}

#if 0 // Can safely remove. Shifted to DFTM
__kernel void DFT(  	__global double *data,
						__global double *scratch,
						const int size, 
						unsigned int dir)
{
	int idX = get_global_id(0);
	double2 TEMP = (double2)(0,0);
	double2 W;

	int i;
	for (i = 0; i < size; i++) 
	{
		W = (double2)( 	cos(CLPT*idX*i/size),
					   -sin(CLPT*idX*i/size));
		TEMP.x += data[2*i] * W.x - data[2*i+1] * W.y;
		TEMP.y += data[2*i] * W.y + data[2*i+1] * W.x;
	}
	scratch[2*idX] 	 	= TEMP.x;
	scratch[2*idX+1] 	= TEMP.y;

	#if 0 // Debug
	data[2*idX] 	= scratch[2*idX];
	data[2*idX+1] 	= scratch[2*idX+1];
	#endif
}
#endif
