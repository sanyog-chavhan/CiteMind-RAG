import os
from dotenv import load_dotenv
from vector_store import get_collection
from config import TOP_K, MAX_OUTPUT_TOKENS, CLAUDE_MODEL
from pathlib import Path
from langchain_anthropic import ChatAnthropic

load_dotenv()

def get_vector_store():
    collection = get_collection()
    return collection

def query_collection(query: str, top_k: int = TOP_K):
    collection = get_vector_store()
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results

def user_query(query: str):
    results = query_collection(query)
    return results

def flatten_results(results):
    flattened = []
    for i in range(len(results['documents'][0])):
        chunk_id = results['ids'][0][i]
        text = results['documents'][0][i]
        page = results['metadatas'][0][i]['page_number']
        source = Path(results['metadatas'][0][i]['file_path']).name
        distance = results['distances'][0][i]
        flattened.append({
            "id": chunk_id,
            "text": text,
            "page": page,
            "source": source,
            "distance": distance
        })
    return flattened

def format_flattened_results(flattened_results):
    formatted = []
    for number, chunk in enumerate(flattened_results, start=1):
        block = f"[{number}] ({chunk['source']}, Page {chunk['page']})\n{chunk['text']}"
        formatted.append(block)
    return "\n\n".join(formatted)

def build_prompt(question: str, context: str) -> str:
    prompt = f"""Answer the question using only the numbered context below.
Cite the sources you use as [1], [2], and so on.
If the context does not contain the answer, say so.

Context:
{context}

Question: {question}
"""
    return prompt

def call_anthropic_api(prompt: str) -> str:
    client = ChatAnthropic(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    response = client.invoke(prompt)
    return response.content

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    results = user_query(user_input)
    flattened_results = flatten_results(results)
    formatted_results = format_flattened_results(flattened_results)
    builded_prompt = build_prompt(user_input, formatted_results)
    response = call_anthropic_api(builded_prompt)
    print("Anthropic API Response:\n")
    print(response)
