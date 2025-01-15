import time
import re

from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# or from langchain.embeddings import FakeEmbeddings  # for quick tests

# 1. Initialize Ollama LLM
# Make sure your Ollama server is running, e.g. "ollama serve -m llama2-7b"
# The default base URL is http://localhost:11411, but you can override with base_url
llm = OllamaLLM(
    model="llama3.1:8b",  
    # base_url="http://172.20.144.1:11434"
)

# 2. Initialize Embeddings & Vector Store
# If you have limited VRAM, you might choose a lightweight embeddings model or use a local approach
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chroma is a straightforward local vector store. 
#   - 'persist_directory' can be set if you want to save the DB on disk.
vector_store = Chroma(
    collection_name="long_term_memory",
    embedding_function=embedder,
    persist_directory="./chroma_data"
)

# Helper functions
def store_memory(text: str):
    """Store the text in our vector database for long-term retrieval."""
    vector_store.add_texts([text])

def retrieve_memories(context: str, k=3):
    """Retrieve the top-k relevant memories from the vector store."""
    docs = vector_store.similarity_search(context, k=k)
    return [doc.page_content for doc in docs]

# Keep track of the ongoing conversation or 'chain of thought'
chat_history = []

def continuous_loop():
    paused = False
    
    # Seed the conversation with a system-style message or context
    chat_history.append("System: You are a continuous-thinking AI with memory.")

    while True:
        if paused:
            user_in = input("(Paused) Provide new input or type 'resume': ")
            if user_in.strip().lower() == 'resume':
                paused = False
                continue
            else:
                chat_history.append(f"User: {user_in}")

        # Retrieve relevant memories based on the most recent message
        last_message = chat_history[-1] if chat_history else ""
        memories = retrieve_memories(last_message, k=3)

        # Compose the prompt for the LLM
        # You can adjust how you integrate these pieces into a single string
        prompt = (
            "System context: You have a continuous train of thought.\n\n"
            f"Recent messages:\n{chat_history[-3:]}\n\n"
            f"Relevant memories:\n{memories}\n\n"
            "Provide your next thought, including any [STORE] or [PAUSE] actions."
        )

        # Generate output from Ollama via LangChain
        # Ollama’s LLM wrapper returns a string (the final text) by default.
        thought = llm(prompt)
        print("AI Thought:", thought)

        # Append to chat history
        chat_history.append(f"AI: {thought}")

        # Check for [STORE] or [PAUSE] directives
        store_tags = re.findall(r'\[STORE\](.*)', thought)
        for st in store_tags:
            store_memory(st.strip())

        if "[PAUSE]" in thought:
            paused = True

        # Sleep to simulate a brief thinking break
        time.sleep(2)

if __name__ == "__main__":
    continuous_loop()
