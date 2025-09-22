# Use Ubuntu 22.04 LTS as the base image
FROM ubuntu:22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Define versions for Go and Apptainer to make updates easier
ENV GO_VERSION=1.22.5
ENV APPTAINER_VERSION=1.3.1

# 1. Install system dependencies
# Install tools required for building Apptainer and for installing uv
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libseccomp-dev \
    pkg-config \
    squashfs-tools \
    cryptsetup \
    uidmap \
    wget \
    git \
    curl \
    python3-pip \
    python3-dev \
    ca-certificates && \
    # Clean up apt cache to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Go
# Go is a prerequisite for compiling Apptainer from source
RUN wget "https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz" && \
    tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz" && \
    rm "go${GO_VERSION}.linux-amd64.tar.gz"

# Add the Go binary directory to the system's PATH
ENV PATH="/usr/local/go/bin:${PATH}"

# 3. Install Apptainer (Singularity)
# Download, compile, and install Apptainer from source
RUN wget "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer-${APPTAINER_VERSION}.tar.gz" && \
    tar -xzf "apptainer-${APPTAINER_VERSION}.tar.gz" && \
    cd "apptainer-${APPTAINER_VERSION}" && \
    ./mconfig && \
    make -C ./builddir && \
    make -C ./builddir install && \
    cd .. && \
    # Clean up source files
    rm -rf "apptainer-${APPTAINER_VERSION}" "apptainer-${APPTAINER_VERSION}.tar.gz"

# 4. Install uv
# Use the official installer script for uv, the fast Python package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add the uv binary directory to the system's PATH
# The installer script places it in /root/.local/bin for the root user
ENV PATH="/root/.local/bin:${PATH}"

# Set a default command to start a bash session when the container runs
CMD ["/bin/bash"]