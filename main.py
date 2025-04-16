import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from code_reader import code_reader
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from prompts import context, code_parser_template

# Load environment variables
load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

# Initialize LLMs
llm = Ollama(model="mistral", request_timeout=120.0)
code_llm = Ollama(model="codellama", request_timeout=60.0)

# Set up PDF parser
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}

# Load documents from ./data folder
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Load embedding model
try:
    embed_model = resolve_embed_model("local:BAAI/bge-m3")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit(1)

# Create vector index and query engine
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = vector_index.as_query_engine(llm=llm)

# Define tools
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="This gives documentation about code for an API. Use this tool for reading API docs.",
        ),
    ),
    code_reader,
]

# Initialize agent
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# Define output structure
class CodeOutput(BaseModel):
    code: str
    description: str
    fileName: str

# Set up output parsing pipeline
parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

# Interactive prompt loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    next_result = output_pipeline.run(response=result)
    print(next_result)
