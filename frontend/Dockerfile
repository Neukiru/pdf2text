FROM python:3.9

COPY . /app
WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt


EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]