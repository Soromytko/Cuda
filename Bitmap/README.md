# Linux
```
nvcc -O2 `pkg-config --cflags opencv4` kernel.cu -o a.out `pkg-config --libs opencv4`
```
