FROM continuumio/miniconda3

WORKDIR /home/biolib

RUN conda install --yes nomkl pytorch scikit-learn pandas numpy && \
    conda clean -afy

COPY . .
