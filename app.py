import streamlit as st
from config import TOP_K
from ingest import ensure_index
from rag import (
    get_vector_store,
    query_collection,
    flatten_results,
    format_flattened_results,
    build_prompt,
    call_anthropic_api,
)

st.set_page_config(page_title="CiteMind", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Building index on first run, this can take a few minutes...")
def load_collection():
    ensure_index()
    return get_vector_store()


def answer_text(content) -> str:
    # ChatAnthropic may return a plain string or a list of content blocks.
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


st.title("CiteMind")
st.caption("Ask a question about Domingos (2012), 'A Few Useful Things to Know about Machine Learning'. Answers cite the retrieved chunks.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=TOP_K)
    show_sources = st.checkbox("Show retrieved chunks", value=True)

load_collection()

question = st.text_input("Your question", placeholder="e.g. What is supervised learning?")

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Retrieving relevant chunks"):
        results = query_collection(question, top_k=top_k)
        flattened = flatten_results(results)

    if not flattened:
        st.warning("No relevant chunks found in the vector store.")
    else:
        context = format_flattened_results(flattened)
        prompt = build_prompt(question, context)

        with st.spinner("Generating answer with Claude"):
            content = call_anthropic_api(prompt)

        st.subheader("Answer")
        st.markdown(answer_text(content))

        if show_sources:
            st.subheader("Retrieved sources")
            for number, chunk in enumerate(flattened, start=1):
                label = f"[{number}] {chunk['source']} — Page {chunk['page']} (distance {chunk['distance']:.3f})"
                with st.expander(label):
                    st.write(chunk["text"])
