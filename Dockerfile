FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONPATH=/swift-api \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai

RUN \
    sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # 基础工具
        curl \
        wget \
        git \
        vim \
        # Python 构建和运行环境 (明确指定版本)
        build-essential \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        # 时区和语言环境
        tzdata \
        locales && \
    \
    # 更新python3的软链接
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    \
    # 设置时区和语言环境
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    locale-gen C.UTF-8 && \
    \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /swift-api

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./requirements.txt .
COPY ./install_all.sh .

RUN sh install_all.sh && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    rm -rf /root/.cache/pip

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import ms_swift; print('Health check passed')" || exit 1

CMD ["/bin/bash"]
