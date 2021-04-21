FROM continuumio/miniconda3:4.9.2-alpine

WORKDIR /home/biolib

COPY submission/tcrpmhc/environment.yml .

RUN conda install --yes pytorch scikit-learn pandas numpy && \
    conda env update -n base --file=environment.yml

COPY . .
