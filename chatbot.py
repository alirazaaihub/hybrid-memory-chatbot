from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

import uuid


# CONFIG
MAX_SHORT_TERM = 5
MAX_TOKENS = 2000
VECTORSTORE_DIR = "db/chroma_user_memory"
TOP_K = 5


# MODELS
GROQ_API_KEY = "YOUR_API_KEY_HERE"  

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    embedding_function=embeddings
)


# STATE
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_message: BaseMessage


# SHORT-TERM MEMORY (SUMMARIZER)
def summarize_node(state: State):
    msgs = state.get("messages", [])

    if not msgs:
        return {
            "messages": [],
            "current_message": None
        }

    current_message = msgs[-1]

   
    if len(msgs) <= MAX_SHORT_TERM:
        return {
            "messages": msgs,
            "current_message": current_message
        }

    last_msgs = msgs[-MAX_SHORT_TERM:]
    old_msgs = msgs[:-MAX_SHORT_TERM]

    summary = llm.invoke(
        [SystemMessage(content=f"Summarize conversation briefly. Max tokens of summary is {MAX_TOKENS}")]
        + old_msgs
    )

    summary_msg = SystemMessage(
        content=f"Summary of previous conversation:\n{summary.content}"
    )

    return {
        "messages": [summary_msg] + last_msgs,
        "current_message": current_message
    }


# LONG-TERM MEMORY SAVE
def save_to_rag(user_id: str, text: str):
    existing = vectorstore.similarity_search(text, k=1)

    if existing and text in existing[0].page_content:
        return

    doc = Document(
        page_content=text,
        metadata={"user_id": user_id, "id": str(uuid.uuid4())}
    )

    vectorstore.add_documents([doc])


# RETRIEVE MEMORY
def retrieve_user_memory(user_id: str, query: str):
    return vectorstore.similarity_search(
        query,
        k=TOP_K,
        filter={"user_id": user_id}
    )


# MEMORY EXTRACTION
def extract_long_term_memory(user_input: str):
    prompt = f"""
Extract ONLY useful long-term memory.

User message:
{user_input}
"""

    result = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    if result == "NONE":
        return None

    return result


# MAIN CHATBOT
def hybrid_chatbot(user_id: str, state: State, user_input: str):

    
    if "messages" not in state:
        state["messages"] = []

    # Add user message
    state["messages"].append(HumanMessage(content=user_input))

   
    updated = summarize_node(state)
    state.update(updated)

    # RAG memory
    retrieved_docs = retrieve_user_memory(user_id, user_input)
    rag_context = "\n".join([doc.page_content for doc in retrieved_docs])

    # STM memory
    stm_context = "\n".join([msg.content for msg in state.get("messages", [])])

    # Final prompt
    final_prompt = f"""
You are an AI assistant with memory.

SHORT TERM:
{stm_context}

LONG TERM:
{rag_context}

User: {user_input}
"""

    response = llm.invoke([SystemMessage(content=final_prompt)])
    answer = response.content.strip()

    # Save AI response
    state["messages"].append(AIMessage(content=answer))

    # Extract memory
    extracted_memory = extract_long_term_memory(user_input)

    if extracted_memory:
        save_to_rag(user_id, f"User memory: {extracted_memory}")

    return state, answer


# RUN
if __name__ == "__main__":
    state = {"messages": []}
    user_id = "user_123"

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        state, reply = hybrid_chatbot(user_id, state, user_input)

        print("AI:", reply)
