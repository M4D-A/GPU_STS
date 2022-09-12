nvcc src/*.cpp src/*.cu -g -G -o gstat;
nvprof ./gstat; 
#rm gstat