FROM tensorflow/serving:2.8.2-gpu

LABEL maintainer="hello@origamidream.me"

# Path to configs
ENV MODEL_CONFIG=${MODEL_BASE_PATH}/models.config
ENV MODEL_CONFIG_POLL_WAIT_SECONDS=60
ENV BATCHING_CONFIG=${MODEL_BASE_PATH}/batching.config

COPY config/models.config ${MODEL_CONFIG}
COPY config/batching.config ${BATCHING_CONFIG}
COPY models/ ${MODEL_BASE_PATH}

EXPOSE 8500

ENTRYPOINT tensorflow_model_server --rest_api_port=8501 \
  --model_config_file=${MODEL_CONFIG} --model_config_file_poll_wait_seconds=${MODEL_CONFIG_POLL_WAIT_SECONDS} \
  --enable_batching --batching_parameters_file=${BATCHING_CONFIG}
