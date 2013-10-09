CC = gcc 
CFLAGS = -Wall -fPIC -O2 
LDFLAGS = -lOpenCL -lm -I/usr/local/cuda/include -g -pg
LDTEST = -lfftw3 -lOpenCL -lTopeFFT -L/opt/topefft -g -pg
CUDAFLAGS = -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft

REQ = src/util.c src/fft1d.c src/fft3d.c src/fft2d.c src/checkers.c src/fft1d_mix.c
OBJ = obj/util.o obj/fft1d.o obj/fft3d.o obj/fft2d.o obj/checkers.o obj/fft1d_mix.o

topeFFT: $(REQ)
	mkdir -p obj
	mkdir -p lib
	mkdir -p bin
	$(CC) $(CFLAGS) -c -o obj/util.o src/util.c $(LDFLAGS)
	$(CC) $(CFLAGS) -c -o obj/fft1d.o src/fft1d.c $(LDFLAGS) 
	$(CC) $(CFLAGS) -c -o obj/fft1d_mix.o src/fft1d_mix.c $(LDFLAGS) 
	$(CC) $(CFLAGS) -c -o obj/fft2d.o src/fft2d.c $(LDFLAGS) 
	$(CC) $(CFLAGS) -c -o obj/fft3d.o src/fft3d.c $(LDFLAGS) 
	$(CC) $(CFLAGS) -c -o obj/checkers.o src/checkers.c $(LDFLAGS) 
	$(CC) -shared -Wl,-soname,libTopeFFT.so.1 -o lib/libTopeFFT.so.1.0 $(OBJ)
	#$(CC) $(CFLAGS) -o bin/topeFFT_cc src/topeFFT_cc.c $(LDFLAGS)
	#bin/topeFFT_cc

tests:
	$(CC) $(CFLAGS) -lrt $(CUDAFLAGS) test/1d.c -o bin/1d $(LDTEST)
	$(CC) $(CFLAGS) -lrt $(CUDAFLAGS) test/1d_mix.c -o bin/1d_mix $(LDTEST)
	$(CC) $(CFLAGS) -lrt $(CUDAFLAGS) test/2d.c -o bin/2d $(LDTEST) 
	$(CC) $(CFLAGS) -lrt $(CUDAFLAGS) test/3d.c -o bin/3d $(LDTEST) 

install:
	mkdir -p /opt/topefft
	cp src/kernels* /opt/topefft
	cp lib/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so.1.0
	ln -sf /opt/topefft/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so.1
	ln -sf /opt/topefft/libTopeFFT.so.1.0 /opt/topefft/libTopeFFT.so

clean:
	rm -f obj/*.o
	rm -f bin/*
	rm -f a.out
	rm -f .sw*
	rm -f .*sw*
	rm -f src/.sw*
	rm -f src/.*
	rm -f src/kern*.ptx
	rm -f lib/*
	rm -f gmon.out	



