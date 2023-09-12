# Facial service

## Installing OpenCV

```shell
sudo apt update
sudo apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
cd opencv; rm -rf build; mkdir build; cd build; cmake -D WITH_GSTREAMER=ON ..; make -j$(nproc); sudo make install
cd ..
```

## Installing Mxnet

```shell
sudo apt update
sudo apt install -y build-essential git ninja-build ccache libopenblas-dev libopencv-dev cmake
cd mxnet
cp config/linux.cmake config.cmake  # or config/linux_gpu.cmake for build with CUDA
rm -rf build
mkdir -p build 
cd build
cmake ..
cmake --build .
cd ..
cd python
pip install .
cd ..
```
