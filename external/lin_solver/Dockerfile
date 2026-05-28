FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-lc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG NSIGHT_COMPUTE_DEB=nsight-compute-2026.1.1_2026.1.1.2-1_amd64.deb
ARG NSIGHT_SYSTEMS_DEB=nsight-systems-2026.2.1_2026.2.1.210-1_amd64.deb
ARG CUDSS_PIP_SPEC=nvidia-cudss-cu12==0.7.1.6
ARG CUDSS_PIP_INSTALL_DEPS=0
ARG CMAKE_PIP_SPEC=cmake==3.31.6
ARG NODE_MAJOR=22
ARG CLAUDE_CODE_NPM_SPEC=@anthropic-ai/claude-code
ARG CODEX_NPM_SPEC=@openai/codex
ARG POWER_PYTHON_PACKAGES=pypower,pandapower,matpower,matpowercaseframes
ARG MATPOWER_REF=master
ARG POWER_NR_CASES=case30,case118,case1197,case_ACTIVSg2000,case3012wp,case6468rte,case8387pegase,case_ACTIVSg25k,case_SyntheticUSA
ARG POWER_NR_DUMP_ITERATION=2
ARG SUITESPARSE_MATRIX_URLS=https://sparse.tamu.edu/MM/Hamm/memplus.tar.gz,https://sparse.tamu.edu/MM/Rajat/rajat27.tar.gz,https://sparse.tamu.edu/MM/Wang/wang3.tar.gz,https://sparse.tamu.edu/MM/ATandT/onetone2.tar.gz,https://sparse.tamu.edu/MM/Rajat/rajat15.tar.gz
ARG CUDA_ARCHITECTURES=86
ARG RELEASE_CFLAGS
ARG RELEASE_CXXFLAGS
ARG RELEASE_FFLAGS
ARG GKLIB_REF=master
ARG METIS_REF=master
ARG PARMETIS_REF=main
ARG SUITESPARSE_REF=v7.12.1
ARG SUPERLU_REF=v7.0.0
ARG SUPERLU_MT_REF=v4.0.2
ARG GLU3_REF=master
ARG STARPU_REF=master
ARG PASTIX_REF=master
ARG STRUMPACK_REF=master
ARG PANGULU_REF=v5.0.0
ARG MUMPS_SUPERBUILD_REF=main

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    THIRD_PARTY_ROOT=/opt/third_party \
    THIRD_PARTY_SRC=/opt/third_party/src \
    THIRD_PARTY_BUILD=/opt/third_party/build \
    THIRD_PARTY_INSTALL=/opt/third_party/install \
    THIRD_PARTY_COMMON_PREFIX=/opt/third_party/install/common \
    THIRD_PARTY_STRUMPACK_CPU_PREFIX=/opt/third_party/install/strumpack-cpu \
    THIRD_PARTY_STRUMPACK_CUDA_PREFIX=/opt/third_party/install/strumpack-cuda \
    THIRD_PARTY_GLU3_PREFIX=/opt/third_party/install/glu3 \
    THIRD_PARTY_PASTIX_CPU_PREFIX=/opt/third_party/install/pastix-cpu \
    THIRD_PARTY_PASTIX_CUDA_PREFIX=/opt/third_party/install/pastix-cuda \
    THIRD_PARTY_PANGULU_PREFIX=/opt/third_party/install/pangulu \
    THIRD_PARTY_MUMPS_PREFIX=/opt/third_party/install/mumps \
    DATASETS_ROOT=/datasets \
    POWER_SYSTEM_DATASET_ROOT=/datasets/power_system \
    MATPOWER_DATASET_ROOT=/datasets/power_system/matpower \
    MATPOWER_MAT_ROOT=/datasets/power_system/matpower_mat \
    POWER_NR_LINEAR_SYSTEM_ROOT=/datasets/power_system/nr_linear_systems \
    BENCHMARK_MATRIX_ROOT=/datasets/benchmark_matrices \
    SUITESPARSE_MATRIX_ROOT=/datasets/benchmark_matrices \
    CUDSS_DIR=/opt/nvidia/cudss \
    MKLROOT=/opt/intel/oneapi/mkl/latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        fd-find \
        file \
        gfortran \
        git \
        gnupg \
        jq \
        less \
        libasound2 \
        libbz2-dev \
        libgmp-dev \
        libgmpxx4ldbl \
        libgomp1 \
        liblapacke-dev \
        liblzma-dev \
        libnuma-dev \
        libopenblas-dev \
        libopenmpi-dev \
        libnss3 \
        libptscotch-dev \
        libscalapack-openmpi-dev \
        libtool \
        libx11-6 \
        libxcb-cursor0 \
        libxkbcommon-x11-0 \
        libxml2-dev \
        libzstd-dev \
        ninja-build \
        numactl \
        pkg-config \
        procps \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        ripgrep \
        rsync \
        texinfo \
        unzip \
        xauth \
        zip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "${CMAKE_PIP_SPEC}" \
    && cmake --version

RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g \
        "${CLAUDE_CODE_NPM_SPEC}" \
        "${CODEX_NPM_SPEC}" \
    && npm cache clean --force \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor > /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        intel-oneapi-mkl-classic-devel \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && mkdir -p /tmp/nsight \
    && cd /tmp/nsight \
    && curl -fsSLO "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/${NSIGHT_COMPUTE_DEB}" \
    && curl -fsSLO "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/${NSIGHT_SYSTEMS_DEB}" \
    && apt-get install -y --no-install-recommends \
        "./${NSIGHT_COMPUTE_DEB}" \
        "./${NSIGHT_SYSTEMS_DEB}" \
    && rm -rf /tmp/nsight /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gettext \
        libtool-bin \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/docker/sparse-env.sh \
     scripts/docker/install_cudss.sh \
     scripts/docker/build_sparse_stack.sh \
     /opt/docker-scripts/

RUN chmod +x /opt/docker-scripts/*.sh

RUN CUDSS_PIP_SPEC="${CUDSS_PIP_SPEC}" \
    CUDSS_PIP_INSTALL_DEPS="${CUDSS_PIP_INSTALL_DEPS}" \
    /opt/docker-scripts/install_cudss.sh

RUN CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    RELEASE_CFLAGS="${RELEASE_CFLAGS}" \
    RELEASE_CXXFLAGS="${RELEASE_CXXFLAGS}" \
    RELEASE_FFLAGS="${RELEASE_FFLAGS}" \
    GKLIB_REF="${GKLIB_REF}" \
    METIS_REF="${METIS_REF}" \
    PARMETIS_REF="${PARMETIS_REF}" \
    SUITESPARSE_REF="${SUITESPARSE_REF}" \
    SUPERLU_REF="${SUPERLU_REF}" \
    SUPERLU_MT_REF="${SUPERLU_MT_REF}" \
    GLU3_REF="${GLU3_REF}" \
    STARPU_REF="${STARPU_REF}" \
    PASTIX_REF="${PASTIX_REF}" \
    STRUMPACK_REF="${STRUMPACK_REF}" \
    PANGULU_REF="${PANGULU_REF}" \
    MUMPS_SUPERBUILD_REF="${MUMPS_SUPERBUILD_REF}" \
    /opt/docker-scripts/build_sparse_stack.sh

COPY CMakeLists.txt /opt/project-tools-src/
COPY src/ /opt/project-tools-src/src/
COPY prepare_datasets/cpp/ /opt/project-tools-src/prepare_datasets/cpp/

RUN cmake -S /opt/project-tools-src -B /opt/project-tools-build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/project-tools-build --target compute_error --parallel "$(nproc)" \
    && cmake --build /opt/project-tools-build --target prepare_dataset_vectors --parallel "$(nproc)" \
    && cp /opt/project-tools-build/prepare_dataset_vectors /usr/local/bin/

COPY scripts/docker/install_power_tools.sh \
     scripts/docker/generate_power_nr_datasets.sh \
     scripts/docker/generate_linear_system_companions.sh \
     scripts/docker/download_suitesparse_matrix.sh \
     /opt/docker-scripts/

COPY prepare_datasets/python/ /opt/prepare-datasets/python/

RUN chmod +x /opt/docker-scripts/*.sh

RUN POWER_PYTHON_PACKAGES="${POWER_PYTHON_PACKAGES}" \
    MATPOWER_REF="${MATPOWER_REF}" \
    /opt/docker-scripts/install_power_tools.sh

RUN POWER_NR_CASES="${POWER_NR_CASES}" \
    POWER_NR_DUMP_ITERATION="${POWER_NR_DUMP_ITERATION}" \
    POWER_CONVERT_SCRIPT_ROOT=/opt/prepare-datasets/python \
    /opt/docker-scripts/generate_power_nr_datasets.sh

RUN SUITESPARSE_MATRIX_URLS="${SUITESPARSE_MATRIX_URLS}" \
    /opt/docker-scripts/download_suitesparse_matrix.sh

RUN /opt/docker-scripts/generate_linear_system_companions.sh

COPY scripts/docker/sparse-env.sh /etc/profile.d/sparse-solvers.sh

ENV PATH=/opt/third_party/install/common/bin:/opt/third_party/install/strumpack-cuda/bin:/opt/third_party/install/strumpack-cpu/bin:/opt/third_party/install/glu3/bin:/opt/third_party/install/pastix-cuda/bin:/opt/third_party/install/pastix-cpu/bin:/opt/third_party/install/pangulu/bin:/opt/third_party/install/mumps/bin:/opt/nvidia/cudss/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/nvidia/cudss/lib:/opt/third_party/install/common/lib:/opt/third_party/install/common/lib64:/opt/third_party/install/strumpack-cuda/lib:/opt/third_party/install/strumpack-cuda/lib64:/opt/third_party/install/strumpack-cpu/lib:/opt/third_party/install/strumpack-cpu/lib64:/opt/third_party/install/glu3/lib:/opt/third_party/install/glu3/lib64:/opt/third_party/install/pastix-cuda/lib:/opt/third_party/install/pastix-cuda/lib64:/opt/third_party/install/pastix-cpu/lib:/opt/third_party/install/pastix-cpu/lib64:/opt/third_party/install/pangulu/lib:/opt/third_party/install/pangulu/lib64:/opt/third_party/install/mumps/lib:/opt/third_party/install/mumps/lib64:/opt/intel/oneapi/mkl/latest/lib:/opt/intel/oneapi/mkl/latest/lib/intel64
ENV CMAKE_PREFIX_PATH=/opt/third_party/install/common:/opt/third_party/install/strumpack-cuda:/opt/third_party/install/strumpack-cpu:/opt/third_party/install/glu3:/opt/third_party/install/pastix-cuda:/opt/third_party/install/pastix-cpu:/opt/third_party/install/pangulu:/opt/third_party/install/mumps:/opt/intel/oneapi:/opt/nvidia/cudss
ENV PKG_CONFIG_PATH=/opt/third_party/install/common/lib/pkgconfig:/opt/third_party/install/common/lib64/pkgconfig:/opt/third_party/install/mumps/lib/pkgconfig
ENV CPATH=/opt/nvidia/cudss/include:/usr/local/cuda/include:/opt/third_party/install/glu3/include:/opt/third_party/install/pangulu/include:/opt/intel/oneapi/mkl/latest/include

WORKDIR /workspace/sparse_direct_solver

COPY . /workspace/sparse_direct_solver

CMD ["/bin/bash"]
