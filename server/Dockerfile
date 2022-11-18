FROM jupyter/datascience-notebook

USER root

RUN echo "Installing dependencies" \
  && apt-get update \
  && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip3 install opencv-python flask

WORKDIR /app

COPY * /app/

CMD ["flask", "--app", "web", "run", "-h", "0.0.0.0"]
