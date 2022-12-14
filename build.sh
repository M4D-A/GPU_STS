nvcc -O3 src/*.cpp src/*.cu -o gstat;
cp gstat /home/adam/NIST-Statistical-Test-Suite/sts;
echo "Compilation done";
echo ""

if [ "$1" = "build" ]; then
    :
elif [ "$1" = "run" ]; then
    time ./gstat;
elif [ "$1" = "debug" ]; then
    cuda-gdb ./gstat;
elif [ "$1" = "nprof" ]; then
    nvprof ./gstat;
elif [ "$1" = "vprof" ]; then
    valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./gstat;
fi

if [ "$2" = "clean" ]; then
    rm gstat;
fi