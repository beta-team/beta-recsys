FROM ubuntu:18.04 AS base

LABEL maintainer="Beta Recsys Project <recsys.beta@gmail.com>"

# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade
RUN apt-get install -y emacs

# Adding wget and bzip2
RUN apt-get install -y wget bzip2 git
RUN apt-get install -y gcc python3-dev

# Add sudo
RUN apt-get -y install sudo

# Add user ubuntu with no password, add to sudo group
RUN adduser --disabled-password --gecos '' ubuntu
RUN adduser ubuntu sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu/
RUN chmod a+rwx /home/ubuntu/

# Anaconda installing
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

# Set path to conda
ENV PATH /home/ubuntu/anaconda3/bin:$PATH

# Updating Anaconda packages
RUN conda update conda
# RUN conda update anaconda
# RUN conda update --all

# Configuring access to Jupyter
RUN mkdir /home/ubuntu/notebooks
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /home/ubuntu/.jupyter/jupyter_notebook_config.py
# jupyter notbook password: root

# clone repo
RUN cd /home/ubuntu && git clone "https://github.com/beta-team/beta-recsys.git" develop

RUN mv /home/ubuntu/develop /home/ubuntu/beta-recsys

RUN cd /home/ubuntu/beta-recsys && pip install --upgrade pip && \
    pip install flake8==3.7.9 --ignore-installed &&\
    pip install --no-cache-dir -r requirements.txt
RUN cd /home/ubuntu/beta-recsys && python setup.py install --record files.txt

# Jupyter listens port: 8888
EXPOSE 8888

CMD ["jupyter", "lab", "--allow-root", "--notebook-dir=/home/ubuntu/beta-recsys", "--ip='0.0.0.0'", "--port=8888", "--no-browser"]