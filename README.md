# minimal_rag

RAG scripts that are meant to be easily reusable. 

Not running on Pachyderm/MLDM meaning that custom images won't be necessary, as long as the environment contains all dependencies needed to run the three scripts.

## Prerequisites

* Access to already deployed embedding and chat models endpoints.
* An environment in which required libraries can be installed

## Steps

* Create a new conda environment in which all required libraries will be installed:
  * conda create -n <ENV_NAME> python=3.10
  * conda activate <ENV_NAME>
  * pip install -r requirements.txt
* Have a list of folders under <INPUT_DIR>, each one containing one or multiple PDF files. Content from each file will be tagged by its folder name.
* Parse and chunk the files under <INPUT_DIR>, within their respective subfolders, and save the resulting list of json files under <PARSED_DOCS_DIR> with:
  * python3 code/parsing.py --input <INPUT_DIR> --output <PARSED_DOCS_DIR> --chunking_strategy by_title --folder_tags
* Embed the chunks from <PARSED_DOCS_DIR> and save them to the Chroma vector database under <VECTORDB_DIR> with:
