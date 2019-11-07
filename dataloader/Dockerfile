FROM python:alpine3.7
COPY . /app
WORKDIR /app
RUN apk add --no-cache --virtual .build-deps gcc musl-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 9000
CMD python ./dataloader.py