Projeto de RAG com LangChain
Este repositório demonstra como utilizar a técnica RAG (Retrieval-Augmented Generation) com a biblioteca LangChain para construir um sistema de consulta a documentos não estruturados, como PDFs, e gerar respostas precisas utilizando modelos de linguagem da OpenAI.

Funcionalidade
Este projeto permite:

Carregar documentos em PDF e extrair seu conteúdo.
Dividir os documentos em partes menores para facilitar o processamento.
Armazenar as partes dos documentos em um vetor de dados para consultas eficientes.
Utilizar a técnica de RAG para gerar respostas contextuais com o modelo GPT-4.
Realizar consultas dinâmicas em documentos específicos, como perguntar sobre dados de faturamento ou informações de interesse dentro de PDFs.
Tecnologias Utilizadas
LangChain: Biblioteca poderosa para trabalhar com modelos de linguagem e integrar fontes de dados.
PyPDFLoader: Para carregar e processar documentos PDF.
OpenAI GPT-4: Modelo de linguagem para gerar respostas precisas.
InMemoryVectorStore: Armazenamento vetorial para guardar e consultar partes de documentos.
Dotenv: Para carregar variáveis de ambiente, como a chave da API da OpenAI.
Como Usar
Instalar as dependências:
bash
Copiar código
%pip install -qU pypdf langchain_community
Configurar a chave da API da OpenAI:
Crie um arquivo .env na raiz do projeto e adicione sua chave de API da OpenAI:

bash
Copiar código
OPENAI_API_KEY=your-api-key-here
Definir o caminho do seu arquivo PDF:
python
Copiar código
file_path = "seu-arquivo.pdf"
Carregar o PDF, processar e gerar respostas:
python
Copiar código
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Carregar o PDF
loader = PyPDFLoader(file_path)
docs = loader.load()

# Inicializar o modelo de linguagem GPT-4
llm = ChatOpenAI(model="gpt-4")

# Dividir o documento em partes menores
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Criar armazenamento vetorial
vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

# Criar retriever para consulta
retriever = vectorstore.as_retriever()

# Carregar o prompt de RAG
prompt = hub.pull("rlm/rag-prompt")

# Configurar a cadeia RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Executar a cadeia para gerar uma resposta
rag_chain.invoke("Qual foi o faturamento das Startups no último ano?")
Como Funciona
Carregamento de documentos: Usamos o PyPDFLoader para carregar o conteúdo do PDF e processá-lo.
Divisão em partes: O documento é dividido em partes menores com o RecursiveCharacterTextSplitter para otimizar o processamento e a recuperação de contexto.
Armazenamento vetorial: As partes divididas são armazenadas no InMemoryVectorStore, o que permite a recuperação eficiente de trechos relevantes do documento.
Consultas: O modelo de linguagem (GPT-4) é usado para gerar respostas precisas a partir dos dados recuperados. A técnica de RAG é aplicada para combinar a recuperação de informações com a geração de texto.
Contribuição
Se você deseja contribuir com este projeto, fique à vontade para enviar pull requests. Sua colaboração é muito bem-vinda!

Licença
Este projeto está licenciado sob a MIT License.
