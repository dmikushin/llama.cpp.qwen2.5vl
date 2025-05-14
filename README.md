# llama.cpp for Qwen 2.5 VL

This is a fork discussed [here](https://github.com/ggml-org/llama.cpp/issues/11483#issuecomment-2727577078).

## Building

```
docker build -f Dockerfile --build-arg CUDA_VERSION=12.6.0 --build-arg CUDA_ARCH=86 -t llama-cpp-qwen2vl:cuda12.6 .
```
