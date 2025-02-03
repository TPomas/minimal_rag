FROM python:3.10

WORKDIR /app

RUN pip install -U pip

RUN pip install unstructured[all-docs]
RUN pip install loguru
RUN pip install chromadb==0.4.24
RUN pip install llama-index==0.10.23
RUN pip install llama-index-embeddings-huggingface==0.1.4
RUN pip install llama-index-vector-stores-chroma==0.1.8
RUN pip install llama-index-llms-openai-like==0.1.3
RUN pip install llama-index-embeddings-openai==0.1.11 
RUN pip install llama-index-llms-openai==0.1.29
RUN pip install streamlit==1.41.0
RUN pip install opencv-python-headless

COPY gui.py /app
COPY rag_schema.py /app
COPY parsing.py /app
COPY embed.py /app
COPY static/ /app/static
COPY .streamlit /app/.streamlit

CMD ["python", "gui.py"]