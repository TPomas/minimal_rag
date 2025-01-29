FROM python:3.10

WORKDIR /app

RUN pip install -U pip

COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install opencv-python-headless
RUN pip install streamlit==1.41.0
RUN python3 -m nltk.downloader punkt

COPY gui.py /app
COPY rag_schema.py /app
COPY parsing.py /app
COPY embed.py /app
COPY static/ /app/static
COPY .streamlit /app/.streamlit

CMD ["python", "gui.py"]
