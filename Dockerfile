FROM python:3.9-alpine

ADD requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
ADD . /app

EXPOSE 8080

