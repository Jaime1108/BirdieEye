FROM tensorflow/tensorflow:latest
#FROM tensorflow/tensorflow:2.12.0
#FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade --ignore-installed blinker
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --upgrade tensorflow==2.20.0 keras==3.12.0
RUN pip install --upgrade tensorflow==2.20.0 keras==3.12.0
# copy application files
RUN mkdir -p /model
COPY . .
#COPY model/test-model.h5 /model/test-model.h5

ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]