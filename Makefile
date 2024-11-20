all: draw.o pgm.o houghGlobal houghConstant houghShared

draw.o:	common/draw.cpp
	g++ -c common/draw.cpp -o ./draw.o `pkg-config --cflags --libs opencv4`

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o

houghGlobal:  houghGlobal.cu pgm.o draw.o
	nvcc houghGlobal.cu pgm.o draw.o -o houghGlobal `pkg-config --cflags --libs opencv4`

houghConstant:  houghConstant.cu pgm.o draw.o
	nvcc houghConstant.cu pgm.o draw.o -o houghConstant `pkg-config --cflags --libs opencv4`

houghShared:  houghShared.cu pgm.o draw.o
	nvcc houghShared.cu pgm.o draw.o -o houghShared `pkg-config --cflags --libs opencv4`
