FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade --ignore-installed blinker
RUN pip install --no-cache-dir -r requirements.txt

# copy application files
COPY . .


ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]