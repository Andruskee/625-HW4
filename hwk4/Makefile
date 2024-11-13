all: detect

multiply: multiply.cu
	nvcc -o multiply -O3 multiply.cu

matrixMul: matrixMul.cu
	nvcc -o matrixMul -O3 matrixMul.cu

detect: bmp.cu detect.cu detectOPT.cu
	nvcc -c -o detectOPT.o -O3 detectOPT.cu
	nvcc -c -o bmp.o -O3 bmp.cu
	nvcc -o detect detect.cu detectOPT.o bmp.o

tests: detectTest

detectTest: board.bmp sunflower.bmp
	#====================  Detect Test  ====================#
	./detect 40 board.bmp boardBASE.bmp boardOPT.bmp
	diff boardBASE.bmp boardOPT.bmp
	./detect 40 sunflower.bmp sunflowerBASE.bmp sunflowerOPT.bmp
	diff sunflowerBASE.bmp sunflowerOPT.bmp

clean:
	rm -f detect multiplyOPT.o bmp.o detectOPT.o
	rm -f sunflowerBASE.bmp sunflowerOPT.bmp
	rm -f boardBASE.bmp boardOPT.bmp
