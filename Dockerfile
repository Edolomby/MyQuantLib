# ==============================================================================
# Multi-stage Dockerfile for MyQuantLib
# Final image: debian:bookworm-slim (~80 MB) with compiled benchmarks
# ==============================================================================

# --------------- Stage 1: Build ---------------
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ \
        cmake \
        make \
        libeigen3-dev \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Boost >= 1.88 required for boost/random/xoshiro.hpp (bookworm ships 1.81)
ARG BOOST_VERSION=1.88.0
ARG BOOST_VERSION_U=1_88_0
RUN wget -qO- \
        "https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_U}.tar.gz" \
    | tar -xz -C /opt \
    && mv "/opt/boost_${BOOST_VERSION_U}" /opt/boost

WORKDIR /src
COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBOOST_ROOT=/opt/boost \
    && cmake --build build -- -j"$(nproc)" \
    && mkdir /benchmarks \
    && find build -maxdepth 1 -type f -executable -exec cp {} /benchmarks/ \;

# --------------- Stage 2: Runtime (minimal) ---------------
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /benchmarks/ /usr/local/bin/

CMD ["EssentialGreekTest"]
