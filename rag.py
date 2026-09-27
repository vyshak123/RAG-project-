from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic


# Step 1: load
pages = PyPDFLoader("pdfs/notes.pdf").load()
print("pages:", len(pages))

# Step 2: split
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)
print("chunks:", len(chunks))

print("--- chunk 10 ---")
print(chunks[10].page_content)
print("--- chunk 11 ---")
print(chunks[11].page_content)
print(chunks[11].metadata["page"])

# Step 3: embeddings
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector = embeddings.embed_query("What is a security governance framework?")
print("vector length:", len(vector))
print("first 5 numbers:", vector[:5])

import shutil
shutil.rmtree("chroma_db", ignore_errors=True) 

# Step 4: store in vector DB
db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("stored", len(chunks), "chunks in the vector database")

# step 5 : vector search 
results = db.similarity_search("What are the main security risks in multi-cloud environments?", k=4)

for r in results:
    print("--- match, page", r.metadata["page"] + 1, "---")
    print(r.page_content[:200])
    print()


# step 6 
load_dotenv()
llm = ChatAnthropic(model="claude-sonnet-5")

response = llm.invoke("Say hello in one sentence")
print("Claude says:", response.content)


#

# step 8 
print("\nPDF chatbot ready. Type 'quit' to exit.\n")

while True:
    question = input("You: ")
    if question.lower() in ("quit", "exit"):
        break

    results = db.similarity_search(question, k=4)
    context = "\n\n".join(r.page_content for r in results)

    prompt = f"""Answer using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the documents."

Context:
{context}

Question: {question}"""

    answer = llm.invoke(prompt)
    print("Bot:", answer.content)
    print()
