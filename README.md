# cuda tensorcore simple example
This code simply compute C = A @ B (@: matrix product) in tensor core.
A,B,C is 16x16 float matrix.
We use nv_bfloat16 for A and B and float for C in tensorcore computation.

# how to execute
```
nvcc -arch=sm_80 main.cu -o out
./out
```
# expected result
```
host tensor-core diff
18480.000000 18480.000000 0.000000 
18755.000000 18755.000000 0.000000 
22660.000000 22660.000000 0.000000 
18825.000000 18825.000000 0.000000 
...(omit)...
504660.000000 504660.000000 0.000000 
511655.000000 511655.000000 0.000000 
580540.000000 580540.000000 0.000000 
515805.000000 515805.000000 0.000000 
******** OK! ********
```
If tensorcore computation is correct, "OK!" is displayed.
