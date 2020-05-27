g++ -E godbolt.cpp -I . -I /opt/cuda/targets/x86_64-linux/include > godbolt.gen.cpp
grep -v '^#' godbolt.gen.cpp > godbolt.gen.final.cpp