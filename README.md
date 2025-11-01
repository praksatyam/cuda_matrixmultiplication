# Matrix Multiplication

Matrix Multiplication is a parallelizable practicle application using CUDA 
cores. This repo is my first time using and desigin hardware speedups for this
application. 

## Thread Organisation

GPU thread organisation is done hierarchically. On the top we have the grid, te grid is subdivided into nultiple blocks, and each block has an index. These blocks contain threads, and the maximum number of threads in each bloc is 1024. 

Blocs can be organised in higher dimensions, so can the threads inside each block. 

Take away is that: 
1) Each bloc has a "volume" of 1024 threads. 

2) The dimensinos are (z,y,x)

How can I check the number of blocks avaliable on my GPU?
'./rundomcode/blocs/cpp'



## GPU Hardware: Coalesced Memory Access





