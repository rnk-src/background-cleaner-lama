FROM --platform=linux/amd64 ubuntu:23.04

RUN apt-get update
RUN apt-get install -y
RUN apt-get install -y tmux
RUN apt-get install -y nano
RUN apt-get install -y build-essential
RUN apt-get install -y rsync
RUN apt-get install -y libgl1
RUN apt-get install -y wget
RUN apt-get install -y curl
RUN apt-get install -y unzip

ARG USERNAME=user
RUN apt-get update
RUN apt-get install -y sudo
RUN useradd -m $USERNAME
RUN echo "$USERNAME:$USERNAME" | chpasswd
RUN usermod -aG sudo $USERNAME
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER $USERNAME:$USERNAME
WORKDIR "/home/$USERNAME"
ENV PATH="/home/$USERNAME/miniconda3/bin:/home/$USERNAME/.local/bin:${PATH}"
ENV PYTHONPATH="/home/$USERNAME/project"

RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN echo "634d76df5e489c44ade4085552b97bebc786d49245ed1a830022b0b406de5817 /tmp/miniconda.sh" > /tmp/miniconda.sh.sha256
RUN sha256sum --check --status < /tmp/miniconda.sh.sha256
RUN bash /tmp/miniconda.sh -bt -p "/home/$USERNAME/miniconda3"
RUN rm /tmp/miniconda.sh
RUN conda build purge
RUN conda init

RUN pip install -U pip
RUN pip install numpy==1.25.0
RUN pip install scipy==1.11.2
RUN pip install torch==2.0.1
RUN pip install torchvision==0.15.2
RUN pip install opencv-python
RUN pip install tensorflow==2.13.0
RUN pip install joblib
RUN pip install matplotlib
RUN pip install pandas
RUN pip install albumentations==0.5.2
RUN pip install pytorch-lightning==2.0.8
RUN pip install tabulate
RUN pip install easydict==1.9.0
RUN pip install kornia==0.5.0
RUN pip install webdataset
RUN pip install packaging
RUN pip install gpustat
RUN pip install tqdm
RUN pip install pyyaml
RUN pip install hydra-core==1.1.0
RUN pip install scikit-learn==1.3.0
RUN pip install tabulate
RUN pip install scikit-image==0.21.0

ENV TORCH_HOME="/home/$USERNAME/.torch"

ADD entrypoint.sh /home/$USERNAME/.local/bin/entrypoint.sh
ENTRYPOINT [ "entrypoint.sh" ]