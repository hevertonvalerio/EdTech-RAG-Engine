import os
# That forces python to ignore warnings berfor load libs.
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
import logging
from dotenv import load_dotenv
from rich.logging import RichHandler
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from prompt import SYSTEM_PROMPT, USER_TEMPLATE
from groq import Groq
import warnings
# ignoring langchain-community deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")


# Load environment variables
load_dotenv()

# Logs Config
logging.basicConfig(
    level=logging.os.getenv("LOGGING"),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("rag_engine")


client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CONFIG = {
    "chroma_path": os.getenv("CHROMA_PATH"),
    "embedding_model": os.getenv("EMBEDDING_MODEL") 
}


def generate_response(query, docs):
    """Use the prompt defined on prompt.py and the user message to send the question to the model, defined on MODEL_NAME."""
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = USER_TEMPLATE.format(question=query, context=context)

    completion = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


def search_knowledge_base(query, k=3):
    """Searching for the most relevant chunks in the vector database"""
    logger.info("Loading the embedding model")
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])

    logger.info("Loading the vector database")
    vector_db = Chroma(persist_directory=CONFIG["chroma_path"], embedding_function=embeddings)

    logger.info(f"Searching on Document for question: [bold cyan]'{query}'[/bold cyan]")

    # similarity_search returns the k pieces of text most similar to the query
    results = vector_db.similarity_search(query, k=k)

    print("\n" + "="*70)
    print("RESULTS FOUND ON VECTOR DATABASE:")
    print("="*70)

    for i, doc in enumerate(results):
        # try to get the number of pages and the origin file on metadata
        page = doc.metadata.get('page', 'N/A')
        source = doc.metadata.get('source', 'unknow').split('\\')[-1] # get just the file name

        print(f"\n[Excerpt {i+1}] pages: {page} | file: {source}")     
        print("-" * 70)
        print(doc.page_content[:200].replace('\n', ' '))
        print("-" * 70)
    
    return results
    

if __name__ == "__main__":

    # 1. Define the topic and question
    topic = input("exemplo: a transição do pensamento concreto para o formal na robótica: ")
    question = f"Crie um quiz de múltipla escolha sobre: {topic}"
    
    # 2. Search on knowledge base and save the results on variable docs
    docs = search_knowledge_base(topic)
    
    # 3. Generate the response with the documents founded on vector database
    response = generate_response(question, docs)
    
    # 4. Print the response
    print("\n" + "="*70)
    print("Take a deep breath and do your best in this answer:")
    print("="*70)
    print(response)

