# windows:
# docker build -t zonal_stats_toolkit:latest . && docker run --rm -it -v "%CD%":/usr/local/wwf_es_beneficiaries zonal_stats_toolkit:latest
# linux/mac:
# docker build -t zonal_stats_toolkit:latest . && docker run --rm -it -v `pwd`:/usr/local/wwf_es_beneficiaries zonal_stats_toolkit:latest
FROM mambaorg/micromamba:1.4.2-bullseye

# We want all RUN commands to use Bash.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Create the environment
RUN micromamba install -y -n base -c conda-forge \
    python=3.11 gdal rasterio fiona geopandas pip
RUN micromamba shell init -s bash -p /opt/conda

# If needed, ensure the file exists and append your activation line
RUN touch /home/mambauser/.bashrc
RUN echo 'micromamba activate base' >> /home/mambauser/.bashrc

USER root
RUN apt-get update -y
RUN apt install git -y
RUN apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    build-essential \
    zlib1g-dev \
    libpython3-dev \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    libgdal-dev

ARG CACHEBUST=3

ARG WORKDIR=/usr/local/wwf_es_beneficiaries
ENV WORKDIR=${WORKDIR}
WORKDIR ${WORKDIR}

COPY requirements.txt /tmp/requirements.txt
RUN micromamba run -n base pip install -r /tmp/requirements.txt

RUN git clone https://github.com/springinnovate/ecoshard.git /usr/local/ecoshard && \
    cd /usr/local/ecoshard && \
    micromamba run -n base pip install --no-build-isolation . && \
    git log -1 --format='%h on %ci' > /usr/local/ecoshard.gitversion

RUN useradd -ms /bin/bash user && chown -R user:user ${WORKDIR} /usr/local/ecoshard
USER user

CMD ["/bin/bash"]