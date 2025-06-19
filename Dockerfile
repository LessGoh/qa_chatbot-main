FROM python:3.11-slim

WORKDIR /

COPY . /

RUN pip install -r req.txt

EXPOSE 8000 8501

CMD ./qa_system.sh
