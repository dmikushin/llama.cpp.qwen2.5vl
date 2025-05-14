ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.4.0
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# CUDA architecture to build for (default is now 8.6)
ARG CUDA_ARCH=86

RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip git libcurl4-openssl-dev libgomp1 ninja-build

WORKDIR /app

COPY . .

# Always use the specified CUDA architecture and build both the CLI and server
RUN cmake -B build \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    -G Ninja . && \
    cmake --build build --config Release --target llama-qwen2vl-cli

RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

# Verify that both the CLI and server executables have been built
RUN if [ ! -f "build/bin/llama-qwen2vl-cli" ] || [ ! -f "build/bin/llama-qwen2vl-server" ]; then \
    echo "llama-qwen2vl executables not found. Aborting."; \
    exit 1; \
    fi

## Base image
FROM ${BASE_CUDA_RUN_CONTAINER} AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

### Full
FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

### Qwen2VL CLI
FROM base AS qwen2vl-cli

COPY --from=build /app/full/llama-qwen2vl-cli /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-qwen2vl-cli" ]

### Qwen2VL Server
FROM base AS qwen2vl-server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-qwen2vl-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/qwen2vl/health" ]

ENTRYPOINT [ "/app/llama-qwen2vl-server" ]
