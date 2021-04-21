FROM continuumio/miniconda3

WORKDIR /home/biolib

COPY submission/tcrpmhc/environment.yml .

RUN conda install --yes nomkl pytorch scikit-learn pandas numpy && \
    conda clean -afy

COPY . .
