#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define CLPI 3.141592653589793238462643383279 // acos(-1)
#define CLPT 6.283185307179586476925286766559 // acos(-1)*2
#define d707 0.707106781186547524400844362104 // cos(acos(-1)/4)

#define wa5 0.3090169944
#define wb5 0.9510565163
#define wc5 0.8090169944
#define wd5 0.5877852523

#define wa7 0.623489801858734
#define wb7 0.78183148246803
#define wc7 0.222520933956314
#define wd7 0.974927912181824
#define we7 0.900968867902419
#define wf7 0.433883739117558

int myPow(int x,int y){
	int i,p=1;
	for(i=0;i<y;i++){
		p*=x;
	}
	return p;
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
	
	#if 1
	int powMaxLvl = 7;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(4.0f,powMaxLvl);
	}
	powX *= pow(4.0f,powRemain);
	powXm1 = powX/4;
	#endif
	#if 0
	int powX = exp2(log2(4.)*stage);
	int powXm1 = powX/4;
	#endif

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

__kernel void DIT8C2C(	__global double *data, 
						__global double2 *twiddle,
						const int size, unsigned int stage,
						unsigned int dir,
						unsigned int useTwiddle) 
 
{
	int idX = get_global_id(0);

	int powMaxLvl = 4;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(8.0f,powMaxLvl);
	}
	powX *= pow(8.0f,powRemain);
	powXm1 = powX/8;

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
						const int facX, const int facY,
						unsigned int stage,
						unsigned int dir,
						unsigned int type) 
{
	#if 0
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	int BASE 	= 0;
	int STRIDE 	= 1;

	#if 1
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int xx;
	for (xx = 0; xx < powLevels; xx++) {
		powX *= pow(2.0f,powMaxLvl);
	}
	powX *= pow(2.0f,powRemain);
	powXm1 = powX/2;
	#endif
	#if 0
	int powX = exp2(log2(2.)*stage);
	int powXm1 = powX/2;
	#endif

	int yIndex, kIndex, clipStart, clipEnd, coeffUse;
	double2 clSet1;
	
	switch(type)
	{
		case 1: BASE 		= idY*x; 
				yIndex 		= idX / powXm1;
				kIndex 		= idX % powXm1;	
				coeffUse 	= kIndex * (x / powX);
				break;
		case 2: BASE 		= idX; 
				STRIDE 		= x; 
				yIndex 		= idY / powXm1;
				kIndex 		= idY % powXm1;
				coeffUse 	= kIndex * (y / powX);
				break;
	}

	clipStart 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX));
	clipEnd 	= 2 * (BASE+STRIDE*(kIndex + yIndex * powX + powXm1));

	clSet1.x 	= cos(two*CLPI*(coeffUse/2)/xR);
	clSet1.y 	= sin(two*CLPI*(coeffUse/2)/xR);

	double4 LOC = (double4)(	data[clipStart + 0],	data[clipStart + 1],
								data[clipEnd + 0],		data[clipEnd + 1]);
	double4 FIN = (double4)(	LOC.x + LOC.z * clSet1.x - LOC.w * clSet1.y,
								LOC.y + LOC.w * clSet1.x + LOC.z * clSet1.y,
								LOC.x - LOC.z * clSet1.x + LOC.w * clSet1.y,
								LOC.y - LOC.w * clSet1.x - LOC.z * clSet1.y);

	data[clipStart + 0] = FIN.x;
	data[clipStart + 1] = FIN.y;
	data[clipEnd + 0] 	= FIN.z;
	data[clipEnd + 1] 	= FIN.w;
	#endif
}

__kernel void DIT7C2C(
			__global double *data,
			const int size,
			unsigned int stage,
			unsigned int dir)
{
#if 1 // correct
	int idX = get_global_id(0);
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(5.0,powMaxLvl);
	}
	powX *= pow(7.0,powRemain);
	powXm1 = powX/7;
	#if 0
	int powX = exp2(log2(5.)*stage);
	int powXm1 = powX/5;
	#endif       
	// x =pow(5.0,2)=25 whereas x=pow(5.0f,2)=24	
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

	int coeffUse = kIndex * (size / powX);
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


__kernel void DIT5C2C(
			__global double *data, 
			const int size,
			unsigned int stage,
			unsigned int dir)
{
#if 1 // correct
	int idX = get_global_id(0);
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(5.0,powMaxLvl);
	}
	powX *= pow(5.0,powRemain);
	powXm1 = powX/5;
	#if 0
	int powX = exp2(log2(5.)*stage);
	int powXm1 = powX/5;
	#endif       
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

	int coeffUse = kIndex * (size / powX);
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

#if 0 //failed attempt1
	int idX = get_global_id(0);
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++) {
		powX *= pow(5.0,powMaxLvl);
	}
	powX *= pow(5.0,powRemain);
	powXm1 = powX/5;
														
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
	double2 clSet1,clSet2, clSet3, clSet4, clSet5,temp2;
	if(stage==1){
		#if 1
		clSet2.x=cos(CLPT*idX/5);
		clSet2.y=-sin(CLPT*idX/5);
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2=TEMPC.x;
		SIG4A.s3=TEMPC.y;

		clSet3.x=cos(CLPT*2*idX/5);
		clSet3.y=-sin(CLPT*2*idX/5);
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4=TEMPC.x;
		SIG4A.s5=TEMPC.y;

		clSet4.x=cos(CLPT*3*idX/5);
		clSet4.y=-sin(CLPT*3*idX/5);
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6=TEMPC.x;
		SIG4A.s7=TEMPC.y;

		clSet5.x=cos(CLPT*4*idX/5);
		clSet5.y=-sin(CLPT*4*idX/5);
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x=TEMPC.x;
		SIG5A.y=TEMPC.y;
		#endif
		
		data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
		
		#if 1
		clSet2.x=cos(CLPT*idX/5);
		clSet2.y=-sin(CLPT*idX/5);
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2=TEMPC.x;
		SIG4A.s3=TEMPC.y;

		clSet3.x=cos(CLPT*2*idX/5);
		clSet3.y=-sin(CLPT*2*idX/5);
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4=TEMPC.x;
		SIG4A.s5=TEMPC.y;

		clSet4.x=cos(CLPT*3*idX/5);
		clSet4.y=-sin(CLPT*3*idX/5);
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6=TEMPC.x;
		SIG4A.s7=TEMPC.y;

		clSet5.x=cos(CLPT*4*idX/5);
		clSet5.y=-sin(CLPT*4*idX/5);
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x=TEMPC.x;
		SIG5A.y=TEMPC.y;
		
		#endif
		data[clipTwo+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipTwo+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
		
		data[clipThr+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipThr+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;

		data[clipFou+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipFou+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
	
		data[clipFiv+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipFiv+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
	}
	else{
		clSet2.x=cos(CLPT*idX/25);
		clSet2.y=-sin(CLPT*idX/25);
		TEMPC.x = SIG4A.s2 * clSet2.x - SIG4A.s3 * clSet2.y;
		TEMPC.y = SIG4A.s2 * clSet2.y + SIG4A.s3 * clSet2.x;
		SIG4A.s2=TEMPC.x;
		SIG4A.s3=TEMPC.y;

		clSet3.x=cos(CLPT*2*idX/25);
		clSet3.y=-sin(CLPT*2*idX/25);
		TEMPC.x = SIG4A.s4 * clSet3.x - SIG4A.s5 * clSet3.y;
		TEMPC.y = SIG4A.s4 * clSet3.y + SIG4A.s5 * clSet3.x;
		SIG4A.s4=TEMPC.x;
		SIG4A.s5=TEMPC.y;

		clSet4.x=cos(CLPT*3*idX/25);
		clSet4.y=-sin(CLPT*3*idX/25);
		TEMPC.x = SIG4A.s6 * clSet4.x - SIG4A.s7 * clSet4.y;
		TEMPC.y = SIG4A.s6 * clSet4.y + SIG4A.s7 * clSet4.x;
		SIG4A.s6=TEMPC.x;
		SIG4A.s7=TEMPC.y;

		clSet5.x=cos(CLPT*4*idX/25);
		clSet5.y=-sin(CLPT*4*idX/25);
		TEMPC.x = SIG5A.x * clSet5.x - SIG5A.y * clSet5.y;
		TEMPC.y = SIG5A.x * clSet5.y + SIG5A.y * clSet5.x;
		SIG5A.x=TEMPC.x;
		SIG5A.y=TEMPC.y;
	
		data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
		data[clipTwo+0] = SIG4A.s0 + (wa5 * SIG4A.s2 + wb5 * SIG4A.s3) 
						+ (-wc5 * SIG4A.s4 + wd5 * SIG4A.s5) 
						+ (-wc5 * SIG4A.s6 - wd5 * SIG4A.s7) 
						+ (wa5 * SIG5A.x - wb5 * SIG5A.y);
		data[clipTwo+1] = SIG4A.s1 + (-wb5 * SIG4A.s2 + wa5 * SIG4A.s3) 
						+ (-wd5 * SIG4A.s4 - wc5 * SIG4A.s5) 
						+ (wd5 * SIG4A.s6 - wc5 * SIG4A.s7) 
						+ (wb5 * SIG5A.x + wa5 * SIG5A.y);
		data[clipThr+0] = SIG4A.s0 + (-wc5 * SIG4A.s2 + wd5 * SIG4A.s3) 
						+ (wa5 * SIG4A.s4 - wb5 * SIG4A.s5) 
						+ (wa5 * SIG4A.s6 + wb5 * SIG4A.s7) 
						+ (-wc5 * SIG5A.x - wd5 * SIG5A.y);
		data[clipThr+1] = SIG4A.s1 + (-wd5 * SIG4A.s2 - wc5 * SIG4A.s3) 
						+ (wb5 * SIG4A.s4 + wa5 * SIG4A.s5) 
						+ (-wb5 * SIG4A.s6 + wa5 * SIG4A.s7) 
						+ (wd5 * SIG5A.x - wc5 * SIG5A.y);
		data[clipFou+0] = SIG4A.s0 + (-wc5 * SIG4A.s2 - wd5 * SIG4A.s3) 
						+ (wa5 * SIG4A.s4 + wb5 * SIG4A.s5) 
						+ (wa5 * SIG4A.s6 - wb5 * SIG4A.s7) 
						+ (-wc5 * SIG5A.x + wd5 * SIG5A.y);
		data[clipFou+1] = SIG4A.s1 + (wd5 * SIG4A.s2 - wc5 * SIG4A.s3) 
						+ (-wb5 * SIG4A.s4 + wa5 * SIG4A.s5) 
						+ (wb5 * SIG4A.s6 + wa5 * SIG4A.s7) 
						+ (-wd5 * SIG5A.x - wc5 * SIG5A.y);
		data[clipFiv+0] = SIG4A.s0 + (wa5 * SIG4A.s2 - wb5 * SIG4A.s3) 
						+ (-wc5 * SIG4A.s4 - wd5 * SIG4A.s5) 
						+ (-wc5 * SIG4A.s6 + wd5 * SIG4A.s7) 
						+ (wa5 * SIG5A.x + wb5 * SIG5A.y);
		data[clipFiv+1] = SIG4A.s1 + (wb5 * SIG4A.s2 + wa5 * SIG4A.s3) 
						+ (wd5 * SIG4A.s4 - wc5 * SIG4A.s5) 
						+ (-wd5 * SIG4A.s6 - wc5 * SIG4A.s7) 
						+ (-wb5 * SIG5A.x + wa5 * SIG5A.y);
	
		#if 0
		data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
		data[clipTwo+0] = SIG4A.s0 + wa5 * (SIG4A.s2 + SIG5A.x) 
						- wc5 * (SIG4A.s4 + SIG4A.s6) 
						+ wb5 * (SIG4A.s3 - SIG5A.y) 
						+ wd5 * (SIG4A.s5 - SIG4A.s7);
		data[clipTwo+1] = SIG4A.s1 + wa5 * (SIG4A.s3 + SIG5A.y) 
						- wc5 * (SIG4A.s5 + SIG4A.s7) 
						- wb5 * (SIG4A.s2 - SIG5A.x) 
						+ wd5 * (SIG4A.s4 - SIG4A.s6);
		data[clipThr+0] = SIG4A.s0 - wc5 * (SIG4A.s2 + SIG5A.x) 
						- wa5 * (SIG4A.s4 + SIG4A.s6) 
						+ wd5 * (SIG4A.s3 - SIG5A.y) 
						- wb5 * (SIG4A.s5 - SIG4A.s7);
		data[clipThr+1] = SIG4A.s1 - wc5 * (SIG4A.s3 + SIG5A.y) 
						- wa5 * (SIG4A.s5 + SIG4A.s7) 
						- wd5 * (SIG4A.s2 - SIG5A.x) 
						- wb5 * (SIG4A.s4 - SIG4A.s6);
		data[clipFou+0] = SIG4A.s0 - wc5 * (SIG4A.s2 + SIG5A.x) 
						- wa5 * (SIG4A.s4 + SIG4A.s6) 
						- wd5 * (SIG4A.s3 - SIG5A.y) 
						- wb5 * (SIG4A.s5 - SIG4A.s7); 
		data[clipFou+1] = SIG4A.s1 - wc5 * (SIG4A.s3 + SIG5A.y) 
						- wa5 * (SIG4A.s5 + SIG4A.s7) 
						+ wd5 * (SIG4A.s2 - SIG5A.x) 
						- wb5 * (SIG4A.s4 - SIG4A.s6);
		data[clipFiv+0] = SIG4A.s0 + wa5 * (SIG4A.s2 + SIG5A.x) 
						- wc5 * (SIG4A.s4 + SIG4A.s6) 
						- wb5 * (SIG4A.s3 - SIG5A.y) 
						+ wd5 * (SIG4A.s5 - SIG4A.s7);
		data[clipFiv+1] = SIG4A.s1 + wa5 * (SIG4A.s3 + SIG5A.y) 
						- wc5 * (SIG4A.s5 + SIG4A.s7) 
						+ wb5 * (SIG4A.s2 - SIG5A.x) 
						+ wd5 * (SIG4A.s4 - SIG4A.s6);
		#endif //correct
	}	
#endif // failed attempt

#if 0 //extra
	#if 0
	clipOne 	= 2*(idX + 0*size/5 ); 	
	clipTwo 	= 2*(idX + 1*size/5 );
	clipThr		= 2*(idX + 2*size/5 );
	clipFou		= 2*(idX + 3*size/5 );
	clipFiv		= 2*(idX + 4*size/5 );
	#endif 
			
	#if 0  //method 1
	double2 p,q,r,s,t,w,wQ,wR,wS,wT;
	double2 tempQ,tempR,tempS,tempT;
	p.x=p.y=0.0;
	q.x=q.y=0.0;
	r.x=r.y=0.0;
	s.x=s.y=0.0;
	t.x=t.y=0.0;		
	int i,N;
	N=size/5;
			
	for(i=0;i<N;i++){
		w.x=cos(CLPT*idX*i/N);
		w.y=-sin(CLPT*idX*i/N);
		clipOne 	= 2*(i + 0*size/5 ); 	
		clipTwo 	= 2*(i + 1*size/5 );
		clipThr		= 2*(i + 2*size/5 );
		clipFou		= 2*(i + 3*size/5 );
		clipFiv		= 2*(i + 4*size/5 );
		p.x+=data[clipOne+0]*w.x - data[clipOne+1]*w.y;
		p.y+=data[clipOne+0]*w.y - data[clipOne+1]*w.x;
				
		q.x+=data[clipTwo+0]*w.x - data[clipTwo+1]*w.y;
		q.y+=data[clipTwo+0]*w.y - data[clipTwo+1]*w.x;

		r.x+=data[clipThr+0]*w.x - data[clipThr+1]*w.y;
		r.y+=data[clipThr+0]*w.y - data[clipThr+1]*w.x;
		
		s.x+=data[clipFou+0]*w.x - data[clipFou+1]*w.y;
		s.y+=data[clipFou+0]*w.y - data[clipFou+1]*w.x;
	
		t.x+=data[clipFiv+0]*w.x - data[clipFiv+1]*w.y;
		t.y+=data[clipFiv+0]*w.y - data[clipFiv+1]*w.x;
	}
		
	N=size;
	wQ.x=cos(CLPT*idX/N); 		wR.x=cos(CLPT*2*idX/N);
	wS.x=cos(CLPT*3*idX/N); 	wT.x=cos(CLPT*4*idX/N);

	wQ.y=-sin(CLPT*idX/N); 		wR.y=-sin(CLPT*2*idX/N);
	wS.y=-sin(CLPT*3*idX/N); 	wS.y=-sin(CLPT*4*idX/N);
	
	out[2*idX] =p.x + (q.x*wQ.x-q.y*wQ.y) + (r.x*wR.x-r.y*wR.y) 
				+ (s.x*wS.x-s.y*wS.y)+(t.x*wT.x-t.y*wT.y);
	out[2*idX+1]=p.y + (q.x*wQ.y+q.y*wQ.x) + (r.x*wR.y+r.y*wR.x) 
				+ (s.x*wS.y+s.y*wS.x)+(t.x*wT.y+t.y*wT.x);
						
	tempQ.x=wQ.x*wa5-wQ.y*(-wb5);
	tempR.x=wR.x*(-wc5)-wR.y*(-wd5);
	tempS.x=wS.x*(-wc5)-wS.y*wd5;
	tempT.x=wT.x*wa5-wT.y*wb5;
			
	tempQ.y=wQ.x*(-wb5)+wQ.y*wa5;
	tempR.y=wR.y*(-wd5)+wR.y*(-wc5);
	tempS.y=wS.x*wd5+wS.y*(-wc5);
	tempT.y=wT.x*wb5+wT.y*wa5;
	
	out[2*(idX+N)]=p.x + (q.x*tempQ.x-q.y*tempQ.y) + (r.x*tempR.x-r.y*tempR.y) 
				+ (s.x*tempS.x-s.y*tempS.y)+(t.x*tempT.x-t.y*tempT.y);
	out[2*(idX+N)+1]=p.y + (q.x*tempQ.y+q.y*tempQ.x) 
				+ (r.x*tempR.y+r.y*tempR.x) + (s.x*tempS.y+s.y*tempS.x)
				+(t.x*tempT.y+t.y*tempT.x);	

	tempQ.x=wQ.x*(-wc5)-wQ.y*(-wd5);
	tempR.x=wR.x*wa5-wR.y*wb5;
	tempS.x=wS.x*wa5-wS.y*(-wb5);
	tempT.x=wT.x*(-wc5)-wT.y*wd5;

	tempQ.y=wQ.x*(-wd5)+wQ.y*(-wc5);
	tempR.y=wR.x*wb5+wR.y*wa5;
	tempS.y=wS.x*(-wb5)+wS.y*wa5;
	tempT.y=wT.x*wd5+wT.y*(-wc5);
	
	out[2*(idX+2*N)]=p.x + (q.x*tempQ.x-q.y*tempQ.y) 
					+ (r.x*tempR.x-r.y*tempR.y) + (s.x*tempS.x-s.y*tempS.y)
					+(t.x*tempT.x-t.y*tempT.y);
	out[2*(idX+2*N)+1]=p.y + (q.x*tempQ.y+q.y*tempQ.x) 
					+ (r.x*tempR.y+r.y*tempR.x) + (s.x*tempS.y+s.y*tempS.x)
					+(t.x*tempT.y+t.y*tempT.x);
	
	tempQ.x=wQ.x*(-wc5)-wQ.y*wd5;
	tempR.x=wR.x*wa5-wR.y*(-wb5);
	tempS.x=wS.x*wa5-wS.y*wb5;
	tempT.x=wT.x*(-wc5)-wT.y*(-wd5);

	tempQ.y=wQ.x*wd5+wQ.y*(-wc5);
	tempR.y=wR.x*(-wb5)+wR.y*wa5;
	tempS.y=wS.x*wb5+wS.y*wa5;
	tempT.y=wT.x*(-wd5)+wT.y*(-wc5);
			
	out[2*(idX+3*N)]=p.x + (q.x*tempQ.x-q.y*tempQ.y) 
					+ (r.x*tempR.x-r.y*tempR.y) + (s.x*tempS.x-s.y*tempS.y)
					+(t.x*tempT.x-t.y*tempT.y);
	out[2*(idX+3*N)+1]=p.y + (q.x*tempQ.y+q.y*tempQ.x) 
					+ (r.x*tempR.y+r.y*tempR.x) + (s.x*tempS.y+s.y*tempS.x)
					+(t.x*tempT.y+t.y*tempT.x);
	
	tempQ.x=wQ.x*wa5-wQ.y*wb5;
	tempR.x=wR.x*(-wc5)-wR.y*wd5;
	tempS.x=wS.x*(-wc5)-wS.y*(-wd5);
	tempT.x=wT.x*wa5-wT.y*(-wb5);

	tempQ.y=wQ.x*wb5+wQ.y*wa5;
	tempR.y=wR.x*wd5+wR.y*(-wc5);
	tempS.y=wS.x*(-wd5)+wS.y*(-wc5);
	tempT.y=wT.x*(-wb5)+wT.y*wa5;
	
	out[2*(idX+4*N)]=p.x + (q.x*tempQ.x-q.y*tempQ.y) 
					+ (r.x*tempR.x-r.y*tempR.y) + (s.x*tempS.x-s.y*tempS.y)
					+(t.x*tempT.x-t.y*tempT.y);
	out[2*(idX+4*N)+1]=p.y + (q.x*tempQ.y+q.y*tempQ.x) 
					+ (r.x*tempR.y+r.y*tempR.x) + (s.x*tempS.y+s.y*tempS.x)
					+(t.x*tempT.y+t.y*tempT.x);
			
	#endif //end method1
	
	#if 0 //step 1  multiply blocks [ ] with respective twiddles

	double2 tw,tw0,tw1,twp,twq,twr,tws,twt,temp;
			
	double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
					data[clipTwo+0],data[clipTwo+1],
					data[clipThr+0],data[clipThr+1],
					data[clipFou+0],data[clipFou+1]);
	double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
	
	int N=size/5;
	
	if(stage==1){
		tw0.x=cos(CLPT*idX/N);
		tw0.y=-sin(CLPT*idX/N);
		temp.x=SIG4A.s0*tw0.x - SIG4A.s1*tw0.y;
		temp.y=SIG4A.s0*tw0.y + SIG4A.s1*tw0.x;
		SIG4A.s0=temp.x;
		SIG4A.s1=temp.y;
				
		tw0.x=cos(CLPT*idX/N);
		tw0.y=-sin(CLPT*idX/N);
		temp.x=SIG4A.s2*tw0.x - SIG4A.s3*tw0.y;
		temp.y=SIG4A.s2*tw0.y + SIG4A.s3*tw0.x;
		SIG4A.s2=temp.x;
		SIG4A.s3=temp.y;
			
		tw0.x=cos(CLPT*idX/N);
		tw0.y=-sin(CLPT*idX/N);
		temp.x=SIG4A.s4*tw0.x - SIG4A.s5*tw0.y;
		temp.y=SIG4A.s4*tw0.y + SIG4A.s5*tw0.x;
		SIG4A.s4=temp.x;
		SIG4A.s5=temp.y;
				
		tw0.x=cos(CLPT*idX/N);
		tw0.y=-sin(CLPT*idX/N);
		temp.x=SIG4A.s6*tw0.x - SIG4A.s7*tw0.y;
		temp.y=SIG4A.s6*tw0.y + SIG4A.s7*tw0.x;
		SIG4A.s6=temp.x;
		SIG4A.s7=temp.y;
				
		tw0.x=cos(CLPT*idX/N);
		tw0.y=-sin(CLPT*idX/N);
		temp.x=SIG5A.x*tw0.x - SIG4A.y*tw0.y;
		temp.y=SIG5A.x*tw0.y + SIG4A.y*tw0.x;
		SIG5A.x=temp.x;
		SIG5A.y=temp.y;
				
		data[clipOne+0]=SIG4A.s0
				
		#if 0		
		tw1.x=tw0.x*tw0.x-tw0.y*tw0.y;
		tw1.y=2*tw0.x*tw0.y;
		
		twp.x=cos(CLPT*idX*0/size);
		twp.y=-sin(CLPT*idX*0/size);
		tw.x= twp.x*tw1.x-twp.y*tw1.y;
		tw.y= twp.x*tw1.y+twp.y*tw1.x;
		temp.x=data[clipOne+0]*tw.x - data[clipOne+1]*tw.y;
		temp.y=data[clipOne+0]*tw.y + data[clipOne+1]*tw.x;
		data[clipOne+0]=temp.x;
		data[clipOne+1]=temp.y;
				
		twq.x=cos(CLPT*idX*1/size);
		twq.y=-sin(CLPT*idX*1/size);
		tw.x= twq.x*tw1.x-twq.y*tw1.y;
		tw.y= twq.x*tw1.y+twq.y*tw1.x;
				
		temp.x=data[clipTwo+0]*tw.x - data[clipTwo+1]*tw.y;
		temp.y=data[clipTwo+0]*tw.y + data[clipTwo+1]*tw.x;
		data[clipTwo+0]=temp.x;
		data[clipTwo+1]=temp.y;
		
		twr.x=cos(CLPT*idX*2/size);
		twr.y=-sin(CLPT*idX*2/size);
		tw.x= twr.x*tw1.x-twr.y*tw1.y;
		tw.y= twr.x*tw1.y+twr.y*tw1.x;
			
		temp.x=data[clipThr+0]*tw.x - data[clipThr+1]*tw.y;
		temp.y=data[clipThr+0]*tw.y + data[clipThr+1]*tw.x;
		data[clipThr+0]=temp.x;
		data[clipThr+1]=temp.y;
				
		tws.x=cos(CLPT*idX*3/size);
		tws.y=-sin(CLPT*idX*3/size);
		tw.x= tws.x*tw1.x-tws.y*tw1.y;
		tw.y= tws.x*tw1.y+tws.y*tw1.x;
				
		temp.x=data[clipFou+0]*tw.x - data[clipFou+1]*tw.y;
		temp.y=data[clipFou+0]*tw.y + data[clipFou+1]*tw.x;
		data[clipFou+0]=temp.x;
		data[clipFou+1]=temp.y;
		
		twt.x=cos(CLPT*idX*4/size);
		twt.y=-sin(CLPT*idX*4/size);
		tw.x= twt.x*tw1.x-twt.y*tw1.y;
		tw.y= twt.x*tw1.y+twt.y*tw1.x;
		
		temp.x=data[clipFiv+0]*tw.x - data[clipFiv+1]*tw.y;
		temp.y=data[clipFiv+0]*tw.y + data[clipFiv+1]*tw.x;
		data[clipFiv+0]=temp.x;
		data[clipFiv+1]=temp.y;
		#endif
	}
	#endif //step 1
			
	#if 0 //step 2  multiply by last twiddle and sum
	else{
		double2 TEMPC;
		double8 SIG4A = (double8)(	data[clipOne+0],data[clipOne+1],
						data[clipTwo+0],data[clipTwo+1],
						data[clipThr+0],data[clipThr+1],
						data[clipFou+0],data[clipFou+1]);
		double2 SIG5A = (double2)(	data[clipFiv+0],data[clipFiv+1]);
			
		data[clipOne+0] = SIG4A.s0 + SIG4A.s2 + SIG4A.s4 + SIG4A.s6 + SIG5A.x;
		data[clipOne+1] = SIG4A.s1 + SIG4A.s3 + SIG4A.s5 + SIG4A.s7 + SIG5A.y;
	
		data[clipTwo+0] = SIG4A.s0 + wa5 * (SIG4A.s2 + SIG5A.x) 
						- wc5 * (SIG4A.s4 + SIG4A.s6) 
						+ wb5 * (SIG4A.s3 - SIG5A.y) 
						+ wd5 * (SIG4A.s5 - SIG4A.s7);
		data[clipTwo+1] = SIG4A.s1 + wa5 * (SIG4A.s3 + SIG5A.y) 
						- wc5 * (SIG4A.s5 + SIG4A.s7) 
						- wb5 * (SIG4A.s2 - SIG5A.x) 
						+ wd5 * (SIG4A.s4 - SIG4A.s6);
		data[clipThr+0] = SIG4A.s0 - wc5 * (SIG4A.s2 + SIG5A.x) 
						- wa5 * (SIG4A.s4 + SIG4A.s6) 
						+ wd5 * (SIG4A.s3 - SIG5A.y) 
						- wb5 * (SIG4A.s5 - SIG4A.s7);
		data[clipThr+1] = SIG4A.s1 - wc5 * (SIG4A.s3 + SIG5A.y) 
						- wa5 * (SIG4A.s5 + SIG4A.s7) 
						- wd5 * (SIG4A.s2 - SIG5A.x) 
						- wb5 * (SIG4A.s4 - SIG4A.s6);
		data[clipFou+0] = SIG4A.s0 - wc5 * (SIG4A.s2 + SIG5A.x) 
						- wa5 * (SIG4A.s4 + SIG4A.s6) 
						- wd5 * (SIG4A.s3 - SIG5A.y) 
						- wb5 * (SIG4A.s5 -SIG4A.s7); 
		data[clipFou+1] = SIG4A.s1 - wc5 * (SIG4A.s3 + SIG5A.y) 
						- wa5 * (SIG4A.s5 + SIG4A.s7) 
						+ wd5 * (SIG4A.s2 - SIG5A.x) 
						- wb5 * (SIG4A.s4 - SIG4A.s6);
		data[clipFiv+0] = SIG4A.s0 + wa5 * (SIG4A.s2 + SIG5A.x) 
						- wc5 * (SIG4A.s4 + SIG4A.s6) 
						- wb5 * (SIG4A.s3 - SIG5A.y) 
						+ wd5 * (SIG4A.s5 - SIG4A.s7);
		data[clipFiv+1] = SIG4A.s1 + wa5 * (SIG4A.s3 + SIG5A.y) 
						- wc5 * (SIG4A.s5 + SIG4A.s7) 
						+ wb5 * (SIG4A.s2 - SIG5A.x) 
						+ wd5 * (SIG4A.s4 - SIG4A.s6);
	}
	#endif //step2	
#endif //extra
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

	int coeffUse = kIndex * (size / powX);
	
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
}

__kernel void DIT3C2CM(	__global double *data,
						const int facX, const int facY,
						unsigned int stage,
						unsigned int dir,
						unsigned int type) 
{
}

__kernel void DIT4C2CM(	__global double *data,
						const int x, const int y,
						unsigned int stage,
						unsigned int dir,
						unsigned int type) 
{
	int idX = get_global_id(0);
	int idY = get_global_id(1);

	#if 1
	#if 1
	int powMaxLvl = 7;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int xx;
	for (xx = 0; xx < powLevels; xx++) {
		powX *= pow(4.0f,powMaxLvl);
	}
	powX *= pow(4.0f,powRemain);
	powXm1 = powX/4;
	#endif
	#if 0
	int powX = exp2(log2(4.)*stage);
	int powXm1 = powX/4;
	#endif

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
	#endif
}

__kernel void DIT5C2CM(	__global double *data,
						const int facX, const int facY,
						unsigned int stage,
						unsigned int dir,
						unsigned int type) 
{
}

__kernel void DIT7C2CM(	__global double *data,
						const int facX, const int facY,
						unsigned int stage,
						unsigned int dir,
						unsigned int type) 
{
}

__kernel void DIT8C2CM(	__global double *data,
						const int facX, const int facY,
						unsigned int stage,
						unsigned int dir,
						unsigned int type) 
{


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
						__global int *bitX,	// bitrev data
						__global int *bitY,
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
				if (idX < bitX[idX]) {
					OLD = 2*(BASE+STRIDE*idX);
					NEW = 2*(BASE+STRIDE*bitX[idX]);
					
					holder = data[NEW];
					data[NEW] = data[OLD];
					data[OLD] = holder;

					holder = data[NEW+1];
					data[NEW+1] = data[OLD+1];
					data[OLD+1] = holder;
				}
				break;
		case 1: BASE = idX; STRIDE = x; 
				if (idY < bitY[idY]) {
					OLD = 2*(BASE+STRIDE*idY);
					NEW = 2*(BASE+STRIDE*bitY[idY]);

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

	#if 1
	int powMaxLvl = 11;
	int powLevels = stage / powMaxLvl;
	int powRemain = stage % powMaxLvl;
	int powX = 1;
	int powXm1 = 1;
	int x;
	for (x = 0; x < powLevels; x++)	powX *= pow(2.0f,powMaxLvl);
	powX *= pow(2.0f,powRemain);
	powXm1 = powX/2;
	#endif
	#if 0
	int powX = exp2(log2(2.)*stage);
	int powXm1 = powX/2;
	#endif

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
		clSet1.x 	= cos(2.*CLPI*(coeffUse/2)/size);
		clSet1.y 	= sin(2.*CLPI*(coeffUse/2)/size);
	}
	if (dir == 0) clSet1.y *= -1;

	double4 LOC = (double4)(	data[clipStart + 0],	data[clipStart + 1],
								data[clipEnd + 0],		data[clipEnd + 1]);
	double4 FIN = (double4)(	LOC.x + LOC.z * clSet1.x - LOC.w * clSet1.y,
								LOC.y + LOC.w * clSet1.x + LOC.z * clSet1.y,
								LOC.x - LOC.z * clSet1.x + LOC.w * clSet1.y,
								LOC.y - LOC.w * clSet1.x - LOC.z * clSet1.y);

	data[clipStart + 0] = FIN.x;
	data[clipStart + 1] = FIN.y;
	data[clipEnd + 0] 	= FIN.z;
	data[clipEnd + 1] 	= FIN.w;
	#if 0	// Debug
	if (quad == 0) {
		[2*idX] = cos(two*CLPI*(coeffUse)/xR);
		[2*idX+1] = -sin(two*CLPI*(coeffUse)/xR);
		[2*xR/2+2*idX] = twiddle[2*coeffUse];
		[2*xR/2+2*idX+1] = twiddle[2*coeffUse+1];
	}
	else if (quad == 1) {
		[2*idX] = cos(two*CLPI*(coeffUse)/xR);
		[2*idX+1] = -sin(two*CLPI*(coeffUse)/xR);
		[2*xR/2+2*idX] = twiddle[2*buad+1];
		[2*xR/2+2*idX+1] = -twiddle[2*buad];
	}
	#endif
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
		tempRev += bitArray[j] * myPow(radix,i);
	}
	bitRev[idX] = tempRev;
	#endif
}

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







