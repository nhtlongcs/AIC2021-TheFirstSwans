# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/
ARG BASE_IMAGE=ubuntu:18.04
ARG PYTHON_VERSION=3.8

# Instal basic utilities
FROM ${BASE_IMAGE} as dev-base
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    gcc \
    libjpeg-dev \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda-installs
ARG PYTHON_VERSION=3.8
ARG CUDA_VERSION=10.2
ARG PYTORCH_VERSION=1.10
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ENV CONDA_OVERRIDE_CUDA=${CUDA_VERSION}
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y \
    python=${PYTHON_VERSION} \
    pytorch=${TORCH_VERSION} torchvision "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/python -m pip install --user mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html && \
    /opt/conda/bin/python -m pip install --user imgaug==0.4.0 mmdet==2.18.1 gdown einops tldextract
# Install mmocr
COPY external/mmocr/ /workspace/mmocr/
WORKDIR /workspace/mmocr
RUN /opt/conda/bin/python -m pip install --user .
# Install packages for transformerocr
COPY external/paddleocr/ /workspace/paddleocr/
WORKDIR /workspace/paddleocr
RUN /opt/conda/bin/python -m pip install --user scikit-learn editdistance


FROM ${BASE_IMAGE} as official
SHELL ["/bin/bash", "-c"]
ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /workspace
ARG PYTORCH_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN --mount=type=cache,id=apt-final,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg libsm6 libxext6 \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*
COPY --from=conda-installs /opt/conda /opt/conda
# copy packages installed by pip
COPY --from=conda-installs /root/.local /root/.local
COPY --from=conda-installs /workspace /workspace
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

RUN /opt/conda/bin/python -m pip install --user pandas nltk 
RUN python -c "import nltk; nltk.download('punkt')"

WORKDIR /workspace
COPY ./mmocr_infer.py ./run.sh /workspace/
COPY ensemble/ /workspace/ensemble
COPY postprocess/ /workspace/postprocess
COPY scripts/ /workspace/scripts
COPY weights/ /workspace/weights

# # ENTRYPOINT ["/bin/bash", "run.sh"]
