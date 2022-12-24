FROM nvidia/cuda:11.3.0-base-ubuntu20.04

RUN apt-get clean && apt-get update && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en_US
ENV LC_ALL en_US.UTF-8

RUN apt-get clean && apt-get update && apt-get install -y iputils-ping \
iperf3 \
net-tools \
dnsutils \
tmux

LABEL maintainer="Y. Yamasaki <yamasaki@hal.ipc.i.u-tokyo.ac.jp>"

SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y tzdata 
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y git \
make \
build-essential \
libssl-dev \
zlib1g-dev \
libbz2-dev \
libreadline-dev \
libsqlite3-dev \
wget \
curl \
llvm \
libncursesw5-dev \
xz-utils \
tk-dev \
libxml2-dev \
libxmlsec1-dev \
libffi-dev \
liblzma-dev \
python3-distutils \
fontconfig
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get update && apt-get install -y --no-install-recommends ttf-mscorefonts-installer

ARG USER
ENV USER=${USER}
RUN useradd -m ${USER}
USER ${USER}
ENV HOME=/home/${USER}
WORKDIR ${HOME}/project
ENTRYPOINT [ "./shell/entrypoint.sh" ]

RUN git clone https://github.com/pyenv/pyenv.git ${HOME}/.pyenv

ENV PYENV_ROOT ${HOME}/.pyenv
ENV PATH ${PYENV_ROOT}/bin:$PATH
ENV PATH ${PYENV_ROOT}/shims:$PATH

RUN eval "$(pyenv init -)" && pyenv install 3.8.0 && pyenv global 3.8.0
RUN pip install pipenv

RUN PIPENV_VENV_IN_PROJECT=1 pipenv install --python 3.8.0 
RUN pipenv install https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl --skip-lock && \
pipenv install https://download.pytorch.org/whl/cu113/torchvision-0.13.0%2Bcu113-cp38-cp38-linux_x86_64.whl --skip-lock && \
pipenv install flwr==1.1.0 --skip-lock && \
pipenv install tqdm --skip-lock && \
pipenv install ray==2.1.0 --skip-lock && \
pipenv install pandas --skip-lock && \
pipenv install optuna --skip-lock && \
pipenv install matplotlib --skip-lock && \
pipenv install wandb --skip-lock && \
pipenv install black && \
pipenv lock