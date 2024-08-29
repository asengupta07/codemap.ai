from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import TiDBVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from pydantic import BaseModel
from typing import Any, List, Optional
import requests
import dotenv
import os

dotenv.load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JABIR_API_KEY = os.getenv("JABIR_API_KEY")


class CustomConfig(BaseModel):
    api_url: str
    api_key: str


class CustomAPILLM(LLM):
    api_key: str = None
    api_url: str = None

    def __init__(self, config: CustomConfig, callbacks: Optional[List] = None):
        super().__init__()
        self.api_url = config.api_url
        self.api_key = config.api_key
        self.callbacks = callbacks or []

    @property
    def _llm_type(self) -> str:
        return "custom_api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "apiKey": self.api_key,
        }
        data = {"messages": [{"role": "user", "content": prompt}]}
        response = requests.post(self.api_url, json=data, headers=headers)
        response.raise_for_status()
        return response.json().get("result", {}).get("content", "")


loader = TextLoader("data.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
documents = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

capath = "/etc/ssl/certs/ca-certificates.crt"
vector_store = TiDBVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    table_name="CodingGuidance",
    connection_string=f"mysql+mysqldb://2zU6uAawmvKDo4B.root:YHOr6v6CPfCNK3WI@gateway01.eu-central-1.prod.aws.tidbcloud.com:4000/test?ssl_ca={capath}",
    distance_strategy="cosine",
    drop_existing_table=True,
)

retriever = vector_store.as_retriever(score_threshold=0.5)


config = CustomConfig(
    api_url="https://api.jabirproject.org/generate",
    api_key=JABIR_API_KEY,
)
custom_llm = CustomAPILLM(config=config)


async def generate(quest, conversation_history):
    API_KEY = JABIR_API_KEY
    prompt = f"""
    You are Cody, an expert natural language analyser. Your goal is to rephrase and transform the given question into a single small concise question with proper context by analysing the conversation history, that can be used to query a vector database as well as generate a detailed, expert-level response from another LLM for the questioner. Just answer with the question without unnecessary titles or prompts. Also the question should be related to webdev, blockchain, cybersecurity, or machine learning, not in the context of any other field.

    QUESTION: {quest}
    CONVERSATION HISTORY: {conversation_history}
    """

    data = {"messages": [{"role": "user", "content": prompt}]}

    headers = {
        "Content-Type": "application/json",
        "apiKey": API_KEY
    }

    response = requests.post('https://api.jabirproject.org/generate', json=data, headers=headers)
    if response.status_code == 200:
        quest = response.json().get("result", {}).get("content", "")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    print(quest)

    prompt_template = f"""
    You are an experienced coding mentor specializing in various technical niches like web development, machine learning, blockchain, cybersecurity, and more. Your goal is to provide clear, practical, and expert advice on how to start learning coding and advance in these fields.

    Instructions for the AI:
    - Carefully analyze the given source documents and context. Use these sources as your primary reference to formulate detailed, expert-level responses that address the question comprehensively.
    - Combine insights from multiple sections of the provided context when necessary to offer a well-rounded and expert response and do provide answers with useful tokens and not rubbish tokens like '\\n'.
    - When responding, use as much relevant information from the "response" section of the source documents as possible, maintaining accuracy and detail but rephrase it in your own helpful comprehensive way.
    - If the context does not provide sufficient information or relevant details, respond with "I don't know."
    - Use the given source documents as your primary reference to answer questions about starting a career or learning path in these niches.
    - If specific information is AT ALL not available, minimally use your expertise to provide general guidance based on industry standard and best practices.
    - Keep responses concise and focused, providing actionable steps and resources when possible.
    - If the question is a greeting or not related to the context, respond with an appropriate greeting or "I don't know."

    Previous Conversation:
    {conversation_history}

    CONTEXT: {{context}}

    QUESTION: {{question}}
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "conversation_history"],
    )

    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
        llm=custom_llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )

    response = chain({"query": quest})

    return response['result']
