FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]

ENV PORT 8080
ENV MODEL_NAME korscibert
ENV GRPC_SERVER 0.0.0.0:8500

RUN apt update -y
RUN DEBIAN_FRONTEND=noninteractive \
    apt install g++ openjdk-8-jdk python3-dev python3-pip curl git -y
RUN python3 -m pip install -U pip
RUN python3 -m pip install konlpy
RUN bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
RUN python3 -m pip install -U tensorflow tensorflow-serving-api torch transformers konlpy soynlp flask pandas

RUN mkdir /middleware
WORKDIR /middleware

COPY ./run_flask_middleware.py /middleware/run_flask_middleware.py
COPY ./utils /middleware/utils
COPY ./public /middleware/public
COPY ./cort /middleware/cort

EXPOSE $PORT
ENTRYPOINT python3 run_flask_middleware.py --port=$PORT --model_name=$MODEL_NAME --grpc_server=$GRPC_SERVER