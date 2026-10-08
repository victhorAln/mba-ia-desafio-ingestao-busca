# Ingestão e Busca Semântica com LangChain e Postgres

Software que lê um PDF, armazena seus trechos como vetores em um PostgreSQL com pgVector e permite fazer perguntas via linha de comando (CLI), respondidas **apenas** com base no conteúdo do PDF.

## Tecnologias

- Python 3 + LangChain
- PostgreSQL + pgVector (via Docker Compose)
- Google Gemini
  - Embeddings: `models/gemini-embedding-001` (3072 dimensões)
  - LLM: `gemini-3.6-flash`

## Estrutura

```
├── docker-compose.yml    # Postgres + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Lê o PDF, divide em chunks, gera embeddings e grava no banco
│   ├── search.py         # Busca os 10 chunks mais relevantes e chama a LLM com o prompt
│   ├── chat.py           # CLI para interação com o usuário
├── document.pdf          # PDF para ingestão
└── README.md
```

## Como funciona

**Ingestão (`src/ingest.py`)**
1. Carrega o `document.pdf` com `PyPDFLoader`.
2. Divide o texto em chunks de **1000 caracteres com overlap de 150** (`RecursiveCharacterTextSplitter`).
3. Gera o embedding de cada chunk (`GoogleGenerativeAIEmbeddings`).
4. Grava os vetores no Postgres com `PGVector`. O envio é feito em lotes de 10 chunks e, se o limite de requisições do plano gratuito do Gemini for atingido (erro 429), o script aguarda 60s e tenta de novo.

Os chunks recebem IDs fixos (`doc-0`, `doc-1`, ...), então rodar a ingestão novamente sobrescreve os registros em vez de duplicá-los.

**Busca e chat (`src/search.py` e `src/chat.py`)**
1. Vetoriza a pergunta do usuário.
2. Busca os **10** resultados mais relevantes com `similarity_search_with_score(query, k=10)`.
3. Concatena os resultados no `CONTEXTO` do prompt e chama a LLM.
4. Exibe a resposta. Perguntas fora do conteúdo do PDF recebem: *"Não tenho informações necessárias para responder sua pergunta."*

## Pré-requisitos

- Python 3
- Docker e Docker Compose
- Uma API Key do Google AI Studio (https://aistudio.google.com/apikey)

## Configuração

1. Crie e ative um ambiente virtual e instale as dependências:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Crie o arquivo `.env` a partir do template e preencha a `GOOGLE_API_KEY`:

```bash
cp .env.example .env
```

| Variável | Descrição | Valor padrão |
|---|---|---|
| `GOOGLE_API_KEY` | API Key do Google Gemini | (preencher) |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings | `models/gemini-embedding-001` |
| `GOOGLE_LLM_MODEL` | Modelo que gera as respostas | `gemini-3.6-flash` |
| `DATABASE_URL` | Conexão com o Postgres | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector | `documento_pdf` |
| `PDF_PATH` | Caminho do PDF a ser ingerido | `document.pdf` |

> **Atenção:** se trocar o modelo de embeddings depois da primeira ingestão, a dimensão dos vetores muda e a ingestão falha. Nesse caso, use outro `PG_VECTOR_COLLECTION_NAME` ou apague o volume do banco (`docker compose down -v`) e rode a ingestão de novo.

## Ordem de execução

Execute todos os comandos a partir da raiz do projeto, com o venv ativado.

1. Subir o banco de dados:

```bash
docker compose up -d
```

2. Executar a ingestão do PDF:

```bash
python src/ingest.py
```

3. Rodar o chat:

```bash
python src/chat.py
```

Digite `sair` (ou pressione `Ctrl+C`) para encerrar.

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da empresa Alfa IA Indústria?
RESPOSTA: O faturamento da empresa Alfa IA Indústria é R$ 548.789.613,65.

PERGUNTA: Em que ano foi fundada a Alfa Saúde LTDA?
RESPOSTA: A empresa Alfa Saúde LTDA foi fundada no ano de 1996.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```
