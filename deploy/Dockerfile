# Reuse the project's pinned CUDA runtime; no separate framework or model plugin.
ARG ASR_BASE_IMAGE=quantatrisk/qwen3-asr@sha256:edce2a10b6b7ef016e6bc92860ff147e4d063527808d2a1f086629569da0ccf8
FROM ${ASR_BASE_IMAGE}
RUN rm -r /app/app
COPY app /app/app
COPY deploy/entrypoint.py /app/deploy/entrypoint.py
COPY deploy/R2T2-LICENSE deploy/R2T2-NOTICE /app/deploy/
WORKDIR /app
ENV PYTHONPATH=/app VLLM_WORKER_MULTIPROC_METHOD=spawn
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=10s --start-period=600s --retries=3 \
    CMD ["python", "/app/deploy/entrypoint.py", "--healthcheck"]
ENTRYPOINT ["python", "/app/deploy/entrypoint.py"]
CMD []
