from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA  
from langchain.prompts import PromptTemplate


# Load retriever (you can build index once using rag_build_index.py)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "data/vector_index",
    embeddings,
    allow_dangerous_deserialization=True #added this true line because of error that FAISS found vector index at data/vector_index,
#but since it was serialized using Python’s pickle format,
#LangChain blocks it by default (for security reasons)
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Smaller/faster model is fine; change if you want
ollama_llm = Ollama(model="mistral")
#ollama_llm = Ollama(model="llama3")

# A crisp, “no hallucination” prompt
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a precise fact-checking assistant. Use ONLY the context to assess the claim.\n"
        "If the context is not enough to decide, answer exactly: INSUFFICIENT.\n\n"
        "Claim: {question}\n\nContext:\n{context}\n\n"
        "Answer in 1-3 sentences, focusing on the claim's veracity and citing context facts. "
        "Do not invent facts.\n\nAnswer:"
    ),
)
def _build_chain():
    return RetrievalQA.from_chain_type(
        llm=ollama_llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=False,
    )

qa_chain = _build_chain()

def _clean_answer(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    # Remove generic openings the LLM sometimes inserts
    for junk in ["I'm not a doctor", "I don't know", "does not relate"]:
        if junk.lower() in t.lower():
            return "INSUFFICIENT"
    # Very short = probably useless
    if len(t) < 15:
        return "INSUFFICIENT"
    return t

def retrieve_evidence(query_text: str) -> str:
    try:
        resp = qa_chain.run(query_text)
        return _clean_answer(resp)
    except Exception:
        return "INSUFFICIENT"
    
#def retrieve_evidence(query_text: str):
 #   """Return factual snippets relevant to the claim."""
  #  chain = RetrievalQA.from_chain_type(llm=ollama_llm, retriever=retriever)
   # response = chain.run(query_text)
    #return response
