# EdTech RAG Engine

Sistema de recuperação de informação e geração de conteúdo pedagógico baseado em RAG (Retrieval-Augmented Generation). O motor consulta documentos técnicos em formato vetorial e utiliza LLMs via API para processar e gerar material de estudo.

## Arquitetura

O sistema é composto por três etapas principais de processamento:

1. **Ingestão e Indexação:** Processamento de documentos PDF para armazenamento em banco vetorial (ChromaDB).
2. **Recuperação (Retrieval):** Busca por similaridade semântica utilizando modelos de *embeddings* (HuggingFace).
3. **Geração (Generation):** Orquestração dos dados recuperados em templates de prompt estruturados e envio para o modelo Llama-3 (Groq API).

## Estrutura do Repositório

* `chroma_db/`: Armazenamento persistente do índice vetorial.
* `query_db.py`: Módulo principal contendo a lógica de busca, orquestração e chamada de API.
* `prompt.py`: Gerenciador de templates de sistema e usuário.
* `.env`: Configurações de ambiente (API Keys e caminhos).

## Configuração e Execução

### Pré-requisitos

* Python 3.10+
* Conta na Groq Cloud com API Key válida.

### Instalação

1. Clonar o repositório.
2. Criar e ativar o ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

```


3. Instalar dependências:
```bash
pip install -r requirements.txt

```



### Variáveis de Ambiente

Crie um arquivo `.env` na raiz com os seguintes parâmetros:

```text
GROQ_API_KEY=sua_chave_aqui
CHROMA_PATH=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

```

### Execução

Para rodar o motor e realizar uma consulta:

```bash
python query_db.py

```

## Roadmap de Desenvolvimento

* [x] Implementação da pipeline RAG e integração com Groq API.
* [x] Modularização de prompts via `prompt.py`.
* [ ] Implementação do módulo de *Quiz Engine* para avaliação didática.
* [ ] Persistência de histórico de interações com o aluno.
* [ ] Desenvolvimento de interface via Streamlit.