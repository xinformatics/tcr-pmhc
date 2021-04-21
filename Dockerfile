FROM python:3.9-slim

WORKDIR /home/biolib

RUN pip install torch scikit-learn pandas numpy

COPY . .

CMD python3 CNN.py
