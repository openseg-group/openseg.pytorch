PYTHON="/root/miniconda3/bin/python"

cd src
/usr/local/cuda-8.0/bin/nvcc -c -o deform_conv_cuda_kernel.cu.so deform_conv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std=c++11

cd cuda

# compile modulated deform conv
/usr/local/cuda-8.0/bin/nvcc -c -o modulated_deform_im2col_cuda.cu.so modulated_deform_im2col_cuda.cu -x cu -Xcompiler -fPIC

# compile deform-psroi-pooling
/usr/local/cuda-8.0/bin/nvcc -c -o deform_psroi_pooling_cuda.cu.so deform_psroi_pooling_cuda.cu -x cu -Xcompiler -fPIC

cd ../..
CC=g++ ${PYTHON} build.py
${PYTHON} build_modulated.py
