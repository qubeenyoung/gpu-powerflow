# ---------- Dockerfile (Optimized Cache & Size) ----------
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. 시스템 기본 패키지 설치 (가장 변경이 적은 부분)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates wget pkg-config \
      python3 python3-pip python3-dev \
      gcc-12 g++-12 cmake ninja-build \
      libeigen3-dev libsuitesparse-dev libopenblas-dev \
      zlib1g-dev tree jq \
    #   libopenmpi-dev openmpi-bin \    나중에 추가?
      && rm -rf /var/lib/apt/lists/*

# 2. 컴파일러 설정
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
      update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# 3. Third-party 라이브러리 빌드 및 설치
WORKDIR /app

# ---- third_party: pybind11 ----
RUN git clone https://github.com/pybind/pybind11.git /app/third_party/pybind11 && \
    cd /app/third_party/pybind11 && git checkout v3.0.1
WORKDIR /app/third_party/pybind11
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      && cmake --build build -j \
      && cmake --install build

# ---- third_party: nlohmann/json ----
RUN git clone https://github.com/nlohmann/json.git /app/third_party/json && \
    cd /app/third_party/json && git checkout v3.12.0
WORKDIR /app/third_party/json
RUN cmake -S . -B build -G Ninja \
      -DJSON_BuildTests=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      && cmake --build build -j \
      && cmake --install build

# ---- third_party: cnpy ----
RUN git clone https://github.com/rogersce/cnpy.git /app/third_party/cnpy
WORKDIR /app/third_party/cnpy
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      && cmake --build build -j \
      && cmake --install build

# ---- third_party: spdlog ----
RUN git clone https://github.com/gabime/spdlog.git /app/third_party/spdlog \
    && cmake -S /app/third_party/spdlog -B /app/third_party/spdlog/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DSPDLOG_BUILD_SHARED=OFF \
        -DSPDLOG_BUILD_EXAMPLE=OFF \
        -DSPDLOG_BUILD_TESTS=OFF \
        -DSPDLOG_BUILD_BENCH=OFF \
        -DSPDLOG_FMT_EXTERNAL=OFF \
    && cmake --build /app/third_party/spdlog/build -j \
    && cmake --install /app/third_party/spdlog/build 

# ---- third_party: AMGX ----
    # thrust도 같이 설치됨
RUN git clone https://github.com/NVIDIA/AMGX.git /app/third_party/AMGX && \
    cd /app/third_party/AMGX && git checkout v2.4.0 && \
    git submodule update --init --recursive
WORKDIR /app/third_party/AMGX
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      && cmake --build build -j \
      && cmake --install build

# ---- dataset: pglib-opf ----
WORKDIR /
RUN mkdir -p /datasets && \
    git clone https://github.com/power-grid-lib/pglib-opf.git /datasets/pglib-opf

WORKDIR /app
RUN rm -rf /app/third_party

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/usr/local
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

WORKDIR /workspace
CMD ["/bin/bash"]