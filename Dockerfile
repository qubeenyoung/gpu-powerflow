# ---------- Dockerfile (Optimized Cache & Size) ----------
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Base system packages (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates wget pkg-config \
      python3 python3-pip python3-dev \
      gcc-12 g++-12 cmake ninja-build \
      libeigen3-dev libsuitesparse-dev libopenblas-dev \
      zlib1g-dev tree jq \
      libcudss0-dev-cuda-12 \
      libsuperlu-dev \
      && rm -rf /var/lib/apt/lists/*

# 2. Compiler selection
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
      update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Common envs
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/usr/local
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV CPATH=/usr/include/libcudss/12

WORKDIR /app

# 3. Third-party libraries
# ---- third_party: pybind11  ----
RUN git clone https://github.com/pybind/pybind11.git /app/third_party/pybind11 && \
    cd /app/third_party/pybind11 && git checkout v3.0.1
WORKDIR /app/third_party/pybind11
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build -j \
    && cmake --install build

# ---- third_party: rapidjson  ----
WORKDIR /app
RUN git clone https://github.com/Tencent/rapidjson.git /app/third_party/rapidjson && \
    cd /app/third_party/rapidjson && git checkout v1.1.0
WORKDIR /app/third_party/rapidjson
RUN cmake -S . -B build -G Ninja \
      -DRAPIDJSON_BUILD_DOC=OFF \
      -DRAPIDJSON_BUILD_EXAMPLES=OFF \
      -DRAPIDJSON_BUILD_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build -j \
    && cmake --install build

# ---- third_party: cnpy ----
WORKDIR /app
RUN git clone https://github.com/rogersce/cnpy.git /app/third_party/cnpy
WORKDIR /app/third_party/cnpy
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build -j \
    && cmake --install build

# ---- third_party: spdlog ----
WORKDIR /app
RUN git clone https://github.com/gabime/spdlog.git /app/third_party/spdlog && \
    cd /app/third_party/spdlog && git checkout v1.17.0 && \
    cmake -S /app/third_party/spdlog -B /app/third_party/spdlog/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DSPDLOG_BUILD_SHARED=ON \
        -DSPDLOG_BUILD_EXAMPLE=OFF \
        -DSPDLOG_BUILD_TESTS=OFF \
        -DSPDLOG_BUILD_BENCH=OFF \
        -DSPDLOG_FMT_EXTERNAL=OFF && \
    cmake --build /app/third_party/spdlog/build -j && \
    cmake --install /app/third_party/spdlog/build

# ---- dataset: pglib-opf (pinned to 23.07) ----
WORKDIR /
RUN mkdir -p /datasets && \
    git clone https://github.com/power-grid-lib/pglib-opf.git /datasets/pglib-opf && \
    cd /datasets/pglib-opf && \
    git checkout v23.07

# Cleanup build sources for third_party to reduce final image size
WORKDIR /app
RUN rm -rf /app/third_party

# 4. Python deps (cache-friendly)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt

# 5. Project source
WORKDIR /app
COPY . .

WORKDIR /workspace
CMD ["/bin/bash"]
