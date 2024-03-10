FROM docker.io/alpine:3.19 AS downloader
WORKDIR /workspace
RUN apk update \
    && apk add --no-cache wget \
    && wget --progress=dot:giga --no-check-certificate https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/ \
    && echo "0c3924556e09ef39d1b6a2338cb61bde  checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" | md5sum -c - \
    && apk del wget \
    && rm -rf /var/cache/apk/* /tmp/* /var/tmp/* /root/.cache

FROM docker.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as final
LABEL maintainer="CyFeng16"
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace
RUN apt-get update \
    && apt-get install --no-install-recommends -y software-properties-common libgl1-mesa-glx \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install --no-install-recommends -y python3.11 python3.11-venv python3.11-dev python3-pip python3.11-distutils \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache
COPY --from=downloader /workspace/checkpoints /workspace/checkpoints/
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install --no-cache-dir -r requirements.txt
COPY . /workspace/
EXPOSE 28439
CMD ["python3", "app.py"]