# RAG

reference source - https://github.com/harishneel1/rag-for-beginners/tree/main
python3 -m venv venv
source venv/bin/activate
pip install langchain langchain-community langchain_text_splitters langchain_openai langchain_chroma python_dotenv langchain_experimental

1. ingestion_pipepline.py -
    1. load all source documents 
    2. chunk it up
    3. embed the chuck
    4. store it in vector database


1. characterTextSplitter, recursiveTextSplitter they are good got FAQ related RAG problems
