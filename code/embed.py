import argparse
import json
import os
import shutil
import chromadb
import torch
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI
from typing import Any, List, Optional, Tuple
#from iso639 import languages
from loguru import logger
Embedding = List[float]




class LengthSafeEmbeddings(OpenAIEmbedding):
    
    def get_embedding(client: OpenAI, text: str, engine: str, **kwargs: Any) -> List[float]:
        """Get embedding.

        NOTE: Copied from OpenAI's embedding utils:
        https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

        Copied here to avoid importing unnecessary dependencies
        like matplotlib, plotly, scipy, sklearn.

        """
        text = text.replace("\n", " ")
        logger.info("length safe")
        if len(text) > 512:
            #split into multiple batches and reassemble after
            embeddings = []
            for i in range(0, len(text), 512):
                embeddings.append(
                    client.embeddings.create(
                        input=[text[i : i + 512]], model=engine, **kwargs
                    ).data[0].embedding
                )
            return embeddings

        return (
            client.embeddings.create(input=[text], model=engine, **kwargs).data[0].embedding
        )


def main(data_path, embed_model, db, model_url=None, translate_model=None, max_tokens=1000, reembed=False):
    collection = db.get_or_create_collection(
        name="documents", metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    docs = []
    index = VectorStoreIndex(
        docs, storage_context=storage_context, embed_model=embed_model
    )
    
    sources = []
    if not args.reembed:
        logger.info("Getting list of sources already available in the DB...")
        chunks = collection.get()
        
        for i in range(len(chunks["ids"])):
            file = chunks["metadatas"][i]["Source"]
            if file not in sources:
                sources.append(file)
    
        logger.info("Sources already embedded:")
        for source in sources:
            logger.info(source)
    
    
    for dirpath, dirs, files in os.walk(data_path):
        for file in files:
            input_file = os.path.join(dirpath, file)

            with open(input_file, "r") as f:
                input_text = json.load(f)
                
                cancel_embedding = False
                
                # Assuming chunks from single json doc all have the same source
                # Check if its source has already been processed
                if not args.reembed:
                    if len(input_text) > 0:
                        source = input_text[0]["metadata"]["source"]
                        if source in sources:
                            cancel_embedding = True
                
                if not cancel_embedding:
                    for doc in input_text:
                        if isinstance(doc, dict):
                            if doc["data_type"] == "Table":
                                text = doc["metadata"]["text_as_html"]
                            else:
                                text = doc["content"]
                            source = doc["metadata"]["source"]
                            if "page_number" in doc["metadata"]:
                                page_number = doc["metadata"]["page_number"]
                            else:
                                page_number = 1
                            if "tag" in doc["metadata"]:
                                tag = doc["metadata"]["tag"]
                            else:
                                tag = ""
                            metadata = {
                                "Source": source,
                                "Page Number": page_number,
                                "Commit": os.environ.get("PACH_JOB_ID", ""),
                                "Tag": tag,
                            }
                            #if model_url and not (doc["metadata"]["lang"] == '' or doc["metadata"]["lang"] == 'en' or doc["metadata"]["lang"] == None) and len(text) > 10:
                            #    llm = OpenAILike(model=translate_model, api_base=model_url, api_key="none")
                            #    generate_kwargs = {
                            #        "temperature": 0.2,
                            #        "top_p": 0.8,
                            #        "max_tokens": max_tokens,
                            #    }
                            #    logger.info(f"generate args {generate_kwargs}")
                            #    lang = languages.get(alpha2=doc["metadata"]["lang"]).name
                            #    logger.info(f'Translating from {doc["metadata"]["lang"]} for {source}')
                            #    translate_template_str_llama3 = f"""
                            #        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
                            #        Please translate the following text from {lang} to English with no extra explanations.
                            #        {text}
                            #        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                            #        """
                            #    translated = llm.complete(translate_template_str_llama3, **generate_kwargs)
                            #    metadata["original"] = text
                            #    metadata["lang"] = doc["metadata"]["lang"]
                            #    text = translated.text

                            docs.append(TextNode(text=text, metadata=metadata))
                            
                else:
                    logger.info(f"Skipping chunks from {input_file}, corresponding to {source}, that have already been embedded")
                    logger.info("Pass the --reembed flag if you are willing to recreate embeddings corresponding to all json files in the data-path directory")

    logger.info("Number of chunks: ", len(docs))

    index.insert_nodes(docs, show_progress=True)
    logger.info("Indexing done!")
    
    # This isn't used and would add (mostly empty) json files to the json documents folder
    #index.storage_context.persist(persist_dir=data_path)
    #logger.info("Persisting done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path-to-db",
        type=str,
        default="db/",
        help="path to chromadb",
    )
    parser.add_argument(
        "--emb-model-path",
        type=str,
        default=None,
        help="Local path or URL to sentence transformer model",
    )
    parser.add_argument(
        "--emb-model-name",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Name of the embedding model",
    )
    parser.add_argument(
        "--data-path", type=str, help="Path to json files with unstructured chunks"
    )
    parser.add_argument(
        "--translate-model", type=str, help="Name of OpenAPI model to translate"
    )
    parser.add_argument(
        "--model-url", type=str, help="URL to OpenAPI model to translate"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1000, help="max_tokens"
    )
    parser.add_argument("--output", help="output directory")
    parser.add_argument("--reembed",
        help="Embed chunks from all json files in the data-path directory, even those already embedded when calling this script previously.",
        action="store_true")
    args = parser.parse_args()
    settings = chromadb.get_settings()
    settings.allow_reset = True
    logger.info(f"Creating/loading db at {args.path_to_db}...")
    db = chromadb.PersistentClient(path=args.path_to_db, settings=settings)
    logger.info("Done!")
    
    if args.emb_model_path.startswith("http"):
        logger.info(f"Using Embedding API model endpoint: {args.emb_model_path}")
        embed_model = LengthSafeEmbeddings(api_base=args.emb_model_path, api_key="dummy", embed_batch_size=32, model_name=args.emb_model_name)
    else:
        logger.info("Loading {}...".format(args.emb_model_path))
        embed_model = HuggingFaceEmbedding(args.emb_model_path)
    main(args.data_path, embed_model, db, args.model_url, args.translate_model, args.max_tokens, args.reembed)
    if args.output:
        shutil.copytree(args.path_to_db)
