import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader # help us read text files, ppts, docx files from a particualr directory
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma # vector database
from dotenv import load_dotenv # to create an .env file with api keys etc

load_dotenv()

def load_documents(docs_path="docs"):
    # load all text files from the docs directory
    print(f'loading documents from {docs_path}')

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f'the directory {docs_path} does not exist')

    loader = DirectoryLoader(
        path=docs_path,
        glob = "*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load() # list of langchain documents

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add documents")

    # for i, doc in enumerate(documents[:2]):
    #     print(f'"\n document {i+1}:')
    #     print(f" source: {doc.metadata['source']}")
    #     print(f" content length:{len(doc.page_content)} characters")
    #     print(f" content preview:{doc.page_content[:100]}..")
    #     print(f" metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    print("splitting documents into chunks..")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    # if chunks:
    #     for i,chunk in enumerate(chunks[:5]):
    #         print(f"-----chunk {i+1}------")
    #         print(f"source: {chunk.metadata['source']}")
    #         print(f"Length: {len(chunk.page_content)} characters")
    #         print(f"Content:")
    #         print(chunk.page_content)
    #         print("-" * 50)
    #     if len(chunks) > 5:
    #         print(f"\n... and {len(chunks) - 5} more chunks")
    return chunks

def create_vectore_store(chunks, persist_directory = "db/chroma_db"):
    print("creating embeddings and storing in chromaDB")
    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")

    # create ChromaDB vector store
    print("-----Creating vector store-----")
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {"hnsw:space":"cosine"}
    )
    print("---finished creating vector store---")
    print(f"vector store created ans saved to {persist_directory}")
    return vectorstore

def main():
    print('main function')

    # 1. loading files
    documents = load_documents(docs_path="docs")

    # 2. chunking the files
    chunks = split_documents(documents)

    # 3. embedding ans storunig in vector db
    vectorstore = create_vectore_store(chunks)


if __name__ == "__main__":
    main()