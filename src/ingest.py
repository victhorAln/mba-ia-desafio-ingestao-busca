import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

REQUIRED_ENV_VARS = (
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL", 
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "PDF_PATH",
)

BATCH_SIZE = 10
MAX_TENTATIVAS = 5
ESPERA_SEGUNDOS = 60


def ingest_pdf():
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            raise RuntimeError(f"Variável de ambiente {var} não definida")

    # 1. Carrega o PDF (um Document por página)
    docs = PyPDFLoader(PDF_PATH).load()

    # 2. Divide em chunks de 1000 caracteres com overlap de 150
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)
    if not splits:
        print("Nenhum conteúdo encontrado no PDF.")
        return

    # Remove metadados vazios, que não agregam na busca
    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    # IDs fixos: rodar a ingestão de novo sobrescreve em vez de duplicar
    ids = [f"doc-{i}" for i in range(len(enriched))]

    # 3. Gera os embeddings e 4. grava no Postgres com pgVector
    embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    # Envia em lotes para respeitar o limite de requisições por minuto do meu  plano gratuito
    for inicio in range(0, len(enriched), BATCH_SIZE):
        fim = inicio + BATCH_SIZE
        for tentativa in range(1, MAX_TENTATIVAS + 1):
            try:
                store.add_documents(documents=enriched[inicio:fim], ids=ids[inicio:fim])
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" not in str(e) or tentativa == MAX_TENTATIVAS:
                    raise
                print(f"Limite de requisições atingido, aguardando {ESPERA_SEGUNDOS}s...")
                time.sleep(ESPERA_SEGUNDOS)
        print(f"Chunks {inicio + 1}-{min(fim, len(enriched))} de {len(enriched)} armazenados.")

    print(f"Ingestão concluída: {len(enriched)} chunks armazenados.")


if __name__ == "__main__":
    ingest_pdf()
