all: pgm.o draw.o hough

draw.o:	common/draw.cpp
	g++ -c common/draw.cpp -o ./draw.o `pkg-config --cflags --libs opencv4`

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o

hough:	houghGlobal.cu pgm.o draw.o
	nvcc houghGlobal.cu pgm.o draw.o -o hough `pkg-config --cflags --libs opencv4`
