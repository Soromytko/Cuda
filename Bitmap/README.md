# Linux
```
nvcc -O2 `pkg-config --cflags opencv4` kernel.cu -o a.out `pkg-config --libs opencv4`
```

# Windows
```
nvcc -I"C:\opencv\build\include" -L"C:\opencv\build\x64\vc16\lib" kernel.cu -o kernel.exe -lopencv_world450
```