FROM continuumio/anaconda3

WORKDIR /home/biolib

COPY submission/tcrpmhc/environment.yml .

# RUN conda env update -n base --file=environment.yml

RUN conda install --yes pytorch scikit-learn pandas numpy

COPY . .
