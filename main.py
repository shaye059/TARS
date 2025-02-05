import time
import re
import sys
import networkx as nx

from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# or from langchain.embeddings import FakeEmbeddings  # for quick tests

# 1. Initialize Ollama LLM
# Make sure your Ollama server is running, e.g. "ollama serve -m llama2-7b"
# The default base URL is http://localhost:11411, but you can override with base_url
llm = OllamaLLM(
    model="llama3.1:8b",  
    base_url="http://localhost:11434"
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

memory_graph = nx.Graph()

# Helper functions
def store_memory(text: str):
    """Store the text in our vector database for long-term retrieval."""
    vector_store.add_texts([text])

def retrieve_memories(context: str, k=3):
    """Retrieve the top-k relevant memories from the vector store."""
    docs = vector_store.similarity_search(context, k=k)
    return [doc.page_content for doc in docs]

def retrieve_memories_advanced(query: str = "", context: str = "", k=3):
    """Retrieve relevant memories based on query or context."""
    if query:
        docs = vector_store.similarity_search(query, k=k)
    else:
        docs = vector_store.similarity_search(context, k=k)
    return [doc.page_content for doc in docs]

def parse_and_handle_retrieval(thought: str):
    """Handle memory retrieval based on AI's requests."""
    matches = re.findall(r'\[RETRIEVE: (.+?)\]', thought)
    if matches:
        results = []
        for query in matches:
            results.extend(retrieve_memories_advanced(query=query))
        return results
    return []


def add_memory_to_graph(text: str):
    """Add memory to graph and connect it to related memories."""
    new_node = len(memory_graph.nodes)  # Unique ID for the memory
    memory_graph.add_node(new_node, content=text)

    # Find related memories
    related_memories = retrieve_memories_advanced(context=text, k=5)
    for related in related_memories:
        for node, data in memory_graph.nodes(data=True):
            if data['content'] == related:
                memory_graph.add_edge(new_node, node)

def connect_and_store_concepts(thought: str):
    """Handle AI's concept formation requests."""
    matches = re.findall(r'\[CONCEPT: (.+?)\]', thought)
    for match in matches:
        related_texts = retrieve_memories_advanced(query=match, k=5)
        merged_concept = " ".join(related_texts)  # Merge related memories
        store_memory(merged_concept)
        add_memory_to_graph(merged_concept)


def reflective_thinking(thought: str):
    """Allow the AI to refine its thoughts recursively."""
    if "[REFLECT]" in thought:
        prompt = (
            f"Original thought: {thought}\n"
            "Reflect on this thought and refine or expand it:"
        )
        refined_thought = llm(prompt)
        return refined_thought
    return thought


def store_larger_concepts(thought: str):
    """Extract and store larger concepts."""
    matches = re.findall(r'\[STORE_CONCEPT: (.+?)\]', thought)
    for match in matches:
        summarized_concept = llm(f"Summarize this concept: {match}")
        store_memory(summarized_concept)


def load_system_prompt(file_path: str) -> str:
    """Load the system prompt from a file."""
    with open(file_path, 'r') as f:
        return f.read()
    

def summarize_history(history: list, llm, keep_last=6) -> list:
    """
    Summarize the chat history, keeping the last `keep_last` responses in full.
    
    Parameters:
        history (list): The full chat history as a list of strings.
        llm: The LLM instance for generating summaries.
        keep_last (int): Number of most recent responses to keep unsummarized.
    
    Returns:
        list: A new history with summarized earlier interactions.
    """
    # Keep the last `keep_last` interactions
    recent_history = history[-keep_last:]
    earlier_history = history[:-keep_last]

    # Combine earlier history for summarization
    earlier_history_text = "\n".join(earlier_history)
    prompt = (
        "Summarize the following conversation in a concise and coherent manner, "
        "preserving important context and details:\n\n" + earlier_history_text
    )

    # Generate summary using the LLM
    summary = llm(prompt)
    
    # Return new history with the summary and recent responses
    return [f"Summary of earlier history: {summary}"] + recent_history

def monitor_and_summarize(history: list, llm, token_limit=32000, keep_last=6):
    """
    Check if the chat history exceeds the token limit and summarize if necessary.

    Parameters:
        history (list): The full chat history as a list of strings.
        llm: The LLM instance for generating summaries.
        token_limit (int): The character limit at which summarization is triggered.
        keep_last (int): Number of most recent responses to keep unsummarized.
    
    Returns:
        list: The updated chat history.
    """
    # Calculate total character length
    total_chars = sum(len(msg) for msg in history)

    if total_chars > token_limit:
        print("History exceeds token limit, summarizing...")
        return summarize_history(history, llm, keep_last=keep_last)
    
    return history



# Keep track of the ongoing conversation or 'chain of thought'
system_prompt_path = "system_prompt.txt"
system_prompt = load_system_prompt(system_prompt_path)
chat_history = []

def continuous_loop():
    paused = False
    
    # Seed the conversation with a system-style message or context
    chat_history.append(f"System: {system_prompt}")

    while True:
        if paused:
            print("(Paused) Provide new input or type 'resume': ")
            user_in = sys.stdin.read()
            if user_in.strip().lower() == 'resume':
                paused = False
                continue
            else:
                chat_history.append(f"Omniscient: {user_in}")

        # Retrieve relevant memories and generate the AI's response
        last_message = chat_history[-1] if chat_history else ""
        memories = retrieve_memories_advanced(context=last_message, k=3)
        prompt = (
            f"Recent messages:\n{chat_history[-6:]}\n"
            f"Relevant memories:\n{memories}\n"
            "Provide your next thought, including [STORE], [RETRIEVE], [CONCEPT], or [PAUSE] actions."
        )

        thought = llm(prompt)
        print(thought)
        chat_history.append(f"AI: {thought}")

        # Monitor and summarize history if it becomes too long
        chat_history = monitor_and_summarize(chat_history, llm)

        # Handle commands (STORE, PAUSE, etc.)
        store_tags = re.findall(r'\[STORE\](.*)', thought)
        for st in store_tags:
            store_memory(st.strip())

        if "[PAUSE]" in thought:
            paused = True

        time.sleep(2)

if __name__ == "__main__":
    continuous_loop()
