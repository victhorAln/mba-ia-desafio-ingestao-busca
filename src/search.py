import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

REQUIRED_ENV_VARS = (
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL",
    "GOOGLE_LLM_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
)


def extrair_texto(response) -> str:
    # O Gemini às vezes devolve .content como lista de blocos em vez de string
    conteudo = response.content
    if isinstance(conteudo, list) and len(conteudo) > 0:
        primeiro = conteudo[0]
        return primeiro.get("text", "") if isinstance(primeiro, dict) else str(primeiro)
    return str(conteudo)


def search_prompt():
    """Monta a busca + LLM e devolve uma função que recebe a pergunta e retorna a resposta.

    Retorna None se a inicialização falhar.
    """
    try:
        for var in REQUIRED_ENV_VARS:
            if not os.getenv(var):
                raise RuntimeError(f"Variável de ambiente {var} não definida")

        embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))

        store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )

        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        model = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"), temperature=0)
        chain = prompt | model
    except Exception as e:
        print(f"Erro ao inicializar a busca: {e}")
        return None

    def responder(pergunta: str) -> str:
        # 1. Vetoriza a pergunta e 2. busca os 10 chunks mais relevantes
        resultados = store.similarity_search_with_score(pergunta, k=10)

        # 3. Concatena os resultados no contexto e chama a LLM
        contexto = "\n\n".join(doc.page_content for doc, _score in resultados)
        resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})

        # 4. Retorna a resposta em texto
        return extrair_texto(resposta).strip()

    return responder
