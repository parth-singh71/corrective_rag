# %% [markdown]
# # Corrective Retrieval-Augmented Generation (CRAG)

# %% [markdown]
# Setting a user-agent environment variable for web requests.

# %%
import os

os.environ["USER_AGENT"] = "CRAG/1.0"

# %%
from typing import Annotated, List
from typing_extensions import TypedDict

# %% [markdown]
# Importing required LangChain modules for handling documents, prompts, and messages.

# %%
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

# %% [markdown]
# Importing graph utilities from LangGraph for workflow automation.

# %%
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# %% [markdown]
# Enums and Pydantic models for structured document grading response.

# %%
from enum import Enum
from pydantic import BaseModel, Field


class RAGDocumentGrade(Enum):
    relevant = "relevant"
    irrelevant = "irrelevant"


class RAGDocumentGraderResponse(BaseModel):
    """Represents the structured response for grading a retrieved document in a RAG-based system."""

    grade: RAGDocumentGrade = Field(
        ...,
        description="""The assigned grade indicating whether the document is relevant ("relevant") or not ("irrelevant")""",
    )
    description: str | None = Field(
        None,
        description="""Additional context or reasoning for the grading, typically provided when the grade is "irrelevant". This field is optional.""",
    )


# %% [markdown]
# Defining the graph state to hold information during execution.


# %%
class CRAGState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    answer: str
    document_grader_response: RAGDocumentGraderResponse
    crawler_response: str
    rag_context: List[Document]


# %% [markdown]
# Setting up memory for tracking graph execution, LLM and embedding function

# %%
memory = MemorySaver()

llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# %% [markdown]
# Defining tools for crawler_agent and binding them to a LLM

# %%
from langchain_core.tools import tool, Tool
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    YouTubeSearchTool,
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool


def get_news_search_tool(region="in-en"):
    news_search_tool_wrapper = DuckDuckGoSearchAPIWrapper(
        region=region,
        time="d",
        max_results=5,
    )
    news_search_tool = Tool(
        name="latest_news_search",
        description="Useful for searching latest news articles.",
        func=DuckDuckGoSearchResults(
            api_wrapper=news_search_tool_wrapper,
            source="news",
        ).run,
    )
    return news_search_tool


def get_web_search_tool():
    news_search_tool = Tool(
        name="web_search",
        description="Useful for searching the web.",
        func=DuckDuckGoSearchResults().run,
    )
    return news_search_tool


wikipedia_tool = Tool(
    name="wikipedia_search",
    description="Useful for searching on Wikipedia.",
    func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
)


wikidata_tool = Tool(
    name="wikidata_search",
    description="Useful for searching on Wikidata.",
    func=WikidataQueryRun(api_wrapper=WikidataAPIWrapper()).run,
)


youtube_search_tool = Tool(
    name="youtube_search",
    description="Useful for searching on youtube.",
    func=YouTubeSearchTool().run,
)


tools = [
    get_web_search_tool(),
    wikipedia_tool,
    wikidata_tool,
    youtube_search_tool,
    get_news_search_tool(),
]
llm_with_tools = llm.bind_tools(tools)

# %% [markdown]
# Setting up Chroma DB for vector storage and retrieval.

# %%
from langchain_chroma import Chroma


chroma_db_collection_name = "rag_db"
chroma_db_path = f"./{chroma_db_collection_name}"
vector_store = Chroma(
    collection_name=chroma_db_collection_name,
    embedding_function=embeddings,
    persist_directory=chroma_db_path,
)

# %% [markdown]
# Functions to extract text from PDFs and store them in Chroma DB.

# %%
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def data_extracter(file_path):
    loader = PyPDFLoader(file_path)
    pages: List[Document] = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages


def split_text(pages: List, chunk_size=1000, chunk_overlap=200):
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = recursive_text_splitter.split_documents(pages)
    return splits


def add_documents_to_db(docs: List[Document]):
    document_ids = vector_store.add_documents(documents=docs)
    return document_ids


def add_document(file_path: str):
    pages = data_extracter(file_path)
    splits = split_text(pages)
    add_documents_to_db(splits)


# %% [markdown]
# Adding some documents to Chroma DB

# %%
from pathlib import Path

folder_path = Path("files")
filenames = [f.name for f in folder_path.iterdir() if f.is_file()]
for f in filenames:
    rel_path = folder_path.joinpath(f)
    add_document(rel_path)
    print("Uploaded:", rel_path)

# %% [markdown]
# ## Graph's Node functions
#
# Function to retrieve relevant documents from Chroma DB based on user query.


# %%
def rag_retriver(state: CRAGState):
    messages = state.get("messages", [])
    last_user_message = None
    n = len(messages)
    for i in range(n - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
    if last_user_message is None:
        raise Exception("No user message found in the conversation.")
    retrieved_docs = vector_store.similarity_search(last_user_message)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    new_state = CRAGState(**state)
    new_state["rag_context"] = retrieved_docs
    new_state["question"] = last_user_message
    new_state["messages"] = [{"content": context, "role": "ai"}]
    return new_state


# %% [markdown]
# Function to grade retrieved documents for relevance.


# %%
def document_grader(state: CRAGState):
    question = state.get("question")
    context = state.get("rag_context", [])
    docs_content = "\n\n".join(doc.page_content for doc in context)

    grader_llm = llm.with_structured_output(RAGDocumentGraderResponse)
    grader_prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "You are an expert evaluator responsible for grading retrieved documents in a Retrieval Augmented Generation (RAG) system. Your task is to assess whether the retrieved context is relevant and useful in answering the question or not, also give a proper reason if the context in not relevant.",
            ),
            ("human", "question: {question}\ncontext: {context}"),
        ]
    )
    chain = grader_prompt_template | grader_llm
    grader_response = chain.invoke({"question": question, "context": docs_content})

    new_state = CRAGState(**state)
    new_state["document_grader_response"] = grader_response
    new_state["messages"] = [
        {"content": grader_response.model_dump_json(), "role": "ai"}
    ]
    return new_state


# %% [markdown]
# Function to rephrase query for web search.


# %%
def rephrase_query(state: CRAGState):
    question = state.get("question")
    prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "You are an expert in query optimization and search enhancement. Your task is to rephrase and improve user query to make them clearer, more specific, and better suited for retrieval in a search engine or a Retrieval Augmented Generation (RAG) system.",
            ),
            ("human", "Question: {question}\nRephrased Question:"),
        ]
    )
    chain = prompt_template | llm | StrOutputParser()
    rephrased_question = chain.invoke({"question": question})

    new_state = CRAGState(**state)
    new_state["question"] = rephrased_question
    new_state["messages"] = [{"content": rephrased_question, "role": "ai"}]
    return new_state


# %% [markdown]
# Function to search for answer using various tools like wikipedia, web search etc.


# %%
def crawler_agent(state: CRAGState):
    if state.get("crawler_response", None) is None:
        llm_input = state.get("question")
    else:
        llm_input = state.get("messages", [])

    response = llm_with_tools.invoke(llm_input)

    new_state = CRAGState(**state)
    new_state["crawler_response"] = response.content
    new_state["messages"] = [response]
    return new_state


# %% [markdown]
# Function to finally respond to the user using the data collected till now.


# %%
def responder(state: CRAGState):
    question = state.get("question")
    final_context = None
    should_consider_rag_context = False
    document_grader_response = state.get("document_grader_response", None)
    if document_grader_response is None or document_grader_response.grade == "relevant":
        should_consider_rag_context = True
    if should_consider_rag_context:
        context = state.get("rag_context", [])
        docs_content = "\n\n".join(doc.page_content for doc in context)
        final_context = docs_content
    else:
        final_context = state.get("crawler_response", "No Context Found")

    prompt_template = ChatPromptTemplate(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.",
            ),
            ("user", "Question: {question}\nContext: {context}\nAnswer:"),
        ]
    )
    chain = prompt_template | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": final_context})

    new_state = CRAGState(**state)
    new_state["answer"] = answer
    new_state["messages"] = [{"content": answer, "role": "assistant"}]
    return new_state


# %% [markdown]
# Creating a graph to structure the RAG pipeline.

# %%
from typing import Literal


def document_grader_route_condition(
    state: CRAGState,
) -> Literal["rephrase_query", "responder"]:
    document_grader_response = state.get("document_grader_response", None)
    if document_grader_response is None or document_grader_response.grade == "relevant":
        return "responder"
    return "rephrase_query"


def custom_tools_condition(
    state: CRAGState,
    messages_key: str = "messages",
) -> Literal["tools", "responder"]:
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "responder"


graph_builder = StateGraph(CRAGState)
graph_builder.add_sequence([rag_retriver, document_grader])
graph_builder.add_node("rephrase_query", rephrase_query)
graph_builder.add_node("crawler_agent", crawler_agent)
graph_builder.add_node("responder", responder)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("rag_retriver")

graph_builder.add_conditional_edges(
    "document_grader",
    document_grader_route_condition,
    path_map={
        "rephrase_query": "rephrase_query",
        "responder": "responder",
    },
)
graph_builder.add_edge("rephrase_query", "crawler_agent")

graph_builder.add_conditional_edges(
    "crawler_agent",
    custom_tools_condition,
    {"tools": "tools", "responder": "responder"},
)
graph_builder.add_edge("tools", "crawler_agent")


graph_builder.add_edge("responder", END)

graph = graph_builder.compile(checkpointer=memory)

# %% [markdown]
# Drawing graph

# %%
graph.get_graph().draw_mermaid()

# %% [markdown]
# Inferencing - Executing the RAG workflow.

# %%
config = {"configurable": {"thread_id": "1"}, "recursion_limit": 25}

# %%
user_input = "what is the name of twitter's ceo?"

initial_state: CRAGState = {
    "messages": [
        {"role": "user", "content": user_input},
    ],
}

events = graph.stream(initial_state, config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()


# %%
user_input = "what is self RAG?"

initial_state: CRAGState = {
    "messages": [
        {"role": "user", "content": user_input},
    ],
}

events = graph.stream(initial_state, config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()


# %%
