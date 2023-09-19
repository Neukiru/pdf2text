FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
  && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip3 install pytesseract
RUN pip3 install opencv-python
RUN pip3 install pillow
RUN python -m pip install --upgrade pip

COPY . /app
WORKDIR /app

RUN apt-get install -y \
    mupdf

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8001" , "--reload"]

