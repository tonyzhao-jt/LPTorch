rm -rf ../lptorch/torch_int/*
cp -r torch-int/torch_int/* ../lptorch/torch_int
mkdir ../lptorch/torch_int/cutlass
cp -r cutlass/include ../lptorch/torch_int/cutlass
cp -r cutlass/tools/util ../lptorch/torch_int/cutlass
