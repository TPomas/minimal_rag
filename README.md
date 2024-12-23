# minimal_rag

RAG scripts that are meant to be easily reusable. 

Not running on Pachyderm/MLDM meaning that custom images won't be necessary, as long as the environment contains all dependencies needed to run the three scripts.

## Prerequisites

* Access to already deployed embedding and chat models endpoints.
* An environment in which required libraries can be installed. For example, here is how you could create a conda environment that is suitable for running the entire solution:
```
conda create -n <ENV_NAME> python=3.10
conda activate <ENV_NAME>
pip install -r requirements.txt
```
* Documents to be used by the RAG solution must be made available under a global <INPUT_DIR> directory and its subdirectories, each one containing one or more pdf files:
```
├── input_dir
│   ├── dir1
│   │   ├── doc1A.pdf
│   │   ├── doc1B.pdf
│   │   ├── ...
│   ├── dir2
│   │   ├── doc2A.pdf
│   │   ├── doc2B.pdf
│   │   ├── ...
│   ├── dir3
│   │   ├── ...
│   ├── ...
```
Document chunks will be tagged by subdirectories name. In the end application, this allows filtering RAG responses to only use one (or more) subdirectory content to provide its answers.

## How to run the solution

* Parse and chunk the files under <INPUT_DIR>, within their respective subfolders, and save the resulting list of json files under <PARSED_DOCS_DIR> with:
```
python3 code/parsing.py --input <INPUT_DIR> --output <PARSED_DOCS_DIR> --chunking_strategy by_title --folder_tags
```
* Embed the chunks from <PARSED_DOCS_DIR> and save them to the Chroma vector database under <VECTORDB_DIR>, using <EMBED_MODEL_PATH> embedding model (path or URL) with:
```
python3 code/embed.py --data-path <PARSED_DOCS_DIR> --emb-model-path <EMBED_MODEL_PATH> --path-to-db <VECTORDB_DIR>
```
**Note:** if not using **BAAI/bge-large-en-v1.5** embedding model, specifying `--emb-model-name <EMBED_MODEL>` is mandatory.
* Start the application using both: embedding model available at <EMBED_MODEL_PATH>, chat model available at <CHAT_MODEL_PATH>, whose name is <CHAT_MODEL_NAME> and vector DB saved under <VECTORDB_DIR>:
```
cd code
streamlit run gui.py -- --path-to-db <VECTORDB_DIR> --model <CHAT_MODEL_NAME> --path-to-chat-model <CHAT_MODEL_PATH> --emb-model-path <EMBED_MODEL_PATH> --cutoff 0.6 --streaming
```
**Notes:** 
* gui.py needs to be called from its own directory, to make content of .streamlit and static directories available to it. Make sure to either take this into account when specifying <VECOTRDB_DIR> and/or to move gui.py, .streamlit and static to another location before running this script.
* Streamlit application will be accessible on port 8501.
## Caveats
* Works with pdf files, but hasn't been tested with other file types, although common ones like .txt and .docx are likely to be supported.
* Need to test by uncommenting "skip_infer_table_types=[]" in parsing script. (running into errors locally)
* To avoid wasting time and resources parsing and embedding documents that have already been processed, the scripts are checking whether existing documents have the same name (except for the .json extension) under <PARSED_DOCS_DIR> (for the parsing step) and whether the vector database already contains chunks from the same initial document (for the embedding step). By default, the scripts will not re-parse documents nor re-embed chunks in those cases. Therefore:
  * Make sure that all source documents have different names, even those from different subdirectories. (although scripts could be updated to bypass this issue)
  * If you are willing to re-parse documents, or re-embed chunks (e.g. in case a document has been updated while keeping the same name, or for testing different parsing parameters and embedding models), you can use the `--reparse` flag with the parsing script and the `reembed`for the embedding script.
