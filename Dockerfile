# ---------- Dockerfile (Optimized Cache & Size) ----------
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG MATLAB_RELEASE=R2026a
ARG MATLAB_PRODUCT_LIST="MATLAB"
ARG MATLAB_INSTALL_LOCATION=/opt/matlab
ARG LICENSE_SERVER
ARG MATPOWER_REF=8.1
ARG CUDSS_PACKAGE_VERSION=0.7.1.4-1
ARG NSIGHT_SYSTEMS_PACKAGE_VERSION=12.8.1-1

# 1. Base system packages (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git ca-certificates wget pkg-config \
      python3 python3-pip python3-dev \
      gcc-12 g++-12 cmake ninja-build \
      libsuitesparse-dev libopenblas-dev \
      libmetis-dev \
      libcudss0-cuda-12=${CUDSS_PACKAGE_VERSION} \
      libcudss0-dev-cuda-12=${CUDSS_PACKAGE_VERSION} \
      cuda-nsight-systems-12-8=${NSIGHT_SYSTEMS_PACKAGE_VERSION} \
      zlib1g-dev tree jq \
      && rm -rf /var/lib/apt/lists/*

# MATLAB runtime dependencies from the MathWorks container reference images.
RUN MATLAB_RELEASE_LOWER="$(printf '%s' "${MATLAB_RELEASE}" | tr '[:upper:]' '[:lower:]')" && \
    wget -q -O /tmp/matlab-deps.txt \
      "https://raw.githubusercontent.com/mathworks-ref-arch/container-images/main/matlab-deps/${MATLAB_RELEASE_LOWER}/ubuntu22.04/base-dependencies.txt" && \
    apt-get update && \
    apt-get install -y --no-install-recommends $(cat /tmp/matlab-deps.txt) && \
    rm -f /tmp/matlab-deps.txt && \
    rm -rf /var/lib/apt/lists/*

# 2. Compiler selection
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
      update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Common envs
ENV CUDA_HOME=/usr/local/cuda
ENV MATLAB_ROOT=${MATLAB_INSTALL_LOCATION}
ENV MATPOWER_HOME=/opt/matpower
ENV MLM_LICENSE_FILE=${LICENSE_SERVER}
ENV PATH=${MATLAB_ROOT}/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=/usr/local
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV CPATH=/usr/include/libcudss/12

WORKDIR /app

# 3. MATLAB
RUN wget -q -O /tmp/mpm https://www.mathworks.com/mpm/glnxa64/mpm && \
    chmod +x /tmp/mpm && \
    HOME=/root /tmp/mpm install \
      --release=${MATLAB_RELEASE} \
      --destination=${MATLAB_INSTALL_LOCATION} \
      --products ${MATLAB_PRODUCT_LIST} \
    || (cat /tmp/mathworks_root.log && false) && \
    rm -f /tmp/mpm /tmp/mathworks_root.log && \
    ln -sf ${MATLAB_INSTALL_LOCATION}/bin/matlab /usr/local/bin/matlab

# 4. Third-party libraries
# ---- third_party: pybind11  ----
RUN git clone https://github.com/pybind/pybind11.git /app/third_party/pybind11 && \
    cd /app/third_party/pybind11 && git checkout v3.0.1
WORKDIR /app/third_party/pybind11
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build -j \
    && cmake --install build

# ---- dataset: pglib-opf (pinned to 23.07) ----
WORKDIR /
RUN mkdir -p /datasets && \
    git clone https://github.com/power-grid-lib/pglib-opf.git /datasets/pglib-opf && \
    cd /datasets/pglib-opf && \
    git checkout v23.07

# ---- dataset: MATPOWER case files ----
RUN rm -rf "${MATPOWER_HOME}" /datasets/matpower && \
    git clone --depth 1 --branch "${MATPOWER_REF}" https://github.com/MATPOWER/matpower.git "${MATPOWER_HOME}" && \
    mkdir -p /datasets/matpower/raw && \
    find "${MATPOWER_HOME}/data" -maxdepth 1 -type f -name 'case*.m' -exec cp {} /datasets/matpower/raw/ \; && \
    test "$(find /datasets/matpower/raw -maxdepth 1 -type f -name 'case*.m' | wc -l)" -eq 78 && \
    { \
      printf 'MATPOWER case datasets\n'; \
      printf 'Source: https://github.com/MATPOWER/matpower.git\n'; \
      printf 'Ref: %s\n' "${MATPOWER_REF}"; \
      printf 'MATPOWER_HOME: %s\n' "${MATPOWER_HOME}"; \
      printf 'MATLAB_ROOT: %s\n' "${MATLAB_ROOT}"; \
      printf 'Stored at: /datasets/matpower/raw\n'; \
      printf 'Copied pattern: data/case*.m\n'; \
      printf 'Case count: 78\n'; \
      find /datasets/matpower/raw -maxdepth 1 -type f -name 'case*.m' -printf '%f\n' | sort; \
    } > /datasets/matpower/MANIFEST.txt && \
    mkdir -p /root/Documents/MATLAB && \
    printf "addpath(genpath('%s'));\n" "${MATPOWER_HOME}" > /root/Documents/MATLAB/startup.m && \
    rm -rf "${MATPOWER_HOME}/.git"

# Cleanup build sources for third_party to reduce final image size
WORKDIR /app
RUN rm -rf /app/third_party

# 5. Python deps (cache-friendly)
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt

# 6. Project source
WORKDIR /app
COPY . .

WORKDIR /workspace
CMD ["/bin/bash"]
