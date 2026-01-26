# Stage 1: Build TA-Lib from source
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    gcc \
    make \
    automake \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source with updated config.guess/config.sub for ARM64
WORKDIR /tmp
RUN curl -L -o ta-lib-0.4.0-src.tar.gz \
    https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && curl -L -o config.guess https://git.savannah.gnu.org/cgit/config.git/plain/config.guess \
    && curl -L -o config.sub https://git.savannah.gnu.org/cgit/config.git/plain/config.sub \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Stage 2: Final runtime image
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib binaries and headers from builder stage
COPY --from=builder /usr/lib /usr/lib
COPY --from=builder /usr/include/ta-lib /usr/include/ta-lib

# Set environment variables for TA-Lib
ENV LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# Install Python dependencies
RUN pip install --no-cache-dir \
    ta-lib==0.4.32 \
    mlflow[extras]==2.21.1 \
    scikit-learn==1.7.1

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code and scripts
COPY bash.sh .
COPY src ./src

# Make bash.sh executable
RUN chmod +x bash.sh

# Use bash.sh as the entrypoint
ENTRYPOINT ["./bash.sh"]