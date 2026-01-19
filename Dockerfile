# Use Ubuntu 22.04 for native ARM64/aarch64 support
FROM ubuntu:22.04

# Prevent interactive prompts and set rendering to headless (CPU-based)
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PATH="/opt/conda/bin:$PATH"

# 1. Install system dependencies required for MuJoCo and rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    cmake \
    libgl1-mesa-dev \
    libosmesa6-dev \
    libglew-dev \
    libglfw3-dev \
    patchelf \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Miniconda for ARM64 (aarch64)
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bf -p /opt/conda \
    && rm /tmp/miniconda.sh

# 3. Give Permission to Anaconda (ToS Acceptance)
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 4. Initialize Conda for the shell
RUN /opt/conda/bin/conda init bash

# 5. Environment Setup
# We use conda-forge to get the MuJoCo binary, avoiding the "MUJOCO_PATH" build error.
RUN /opt/conda/bin/conda create -n robomimic_venv python=3.9 -y && \
    /opt/conda/bin/conda install -n robomimic_venv -c conda-forge mujoco -y && \
    /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir \
    pip setuptools wheel "numpy<2.0" && \
    /opt/conda/bin/conda run -n robomimic_venv pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Install robosuite and robomimic from source
WORKDIR /opt

RUN git clone https://github.com/ARISE-Initiative/robosuite.git && \
    /opt/conda/bin/conda run -n robomimic_venv pip install -e ./robosuite

RUN git clone https://github.com/ARISE-Initiative/robomimic.git && \
    /opt/conda/bin/conda run -n robomimic_venv pip install -e ./robomimic

# Final setup
WORKDIR /workspace
RUN /opt/conda/bin/conda clean -ya

# Automatically activate environment on startup
ENTRYPOINT ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "robomimic_venv"]
CMD ["bash"]