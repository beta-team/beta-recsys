ARG BASE_IMAGE="ubuntu:18.04"

FROM ${BASE_IMAGE}

LABEL maintainer="Beta Recsys Project <recsys.beta@gmail.com>"

WORKDIR /root

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y emacs

# Adding wget and bzip2
RUN apt-get install -y wget bzip2
RUN apt-get install -y gcc python3-dev

# Anaconda installing
# Mirror: https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

# Set path to conda
ENV PATH /root/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
# RUN conda update anaconda
# RUN conda update --all

# Configuring access to Jupyter
RUN mkdir /root/notebooks
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip='*'" >> /root/.jupyter/jupyter_notebook_config.py
# jupyter notbook password: root

# clone repo
RUN mkdir /root/beta-recsys
ADD . /root/beta-recsys

RUN cd /root/beta-recsys && pip install --upgrade pip && \
    pip install jupyterlab && \
    pip install flake8==3.7.9 --ignore-installed &&\
    pip install --no-cache-dir -r requirements.txt
RUN cd /root/beta-recsys && python setup.py install --record files.txt

# Jupyter listens port: 8888
EXPOSE 8888

CMD ["jupyter", "lab", "--allow-root", "--notebook-dir=/root/beta-recsys", "--ip='*'", "--port=8888", "--no-browser"]