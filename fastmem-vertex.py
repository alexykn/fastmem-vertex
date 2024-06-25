import os
import logging
import time
import json
from dotenv import load_dotenv
from anthropic import AnthropicVertex
import anthropic.types
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from langchain_community.chat_message_histories import ChatMessageHistory as ChatMessageHistory
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
import faiss

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get environment variables
PROJECT_ID = os.getenv("ANTHROPIC_PROJECT_ID")
LOCATION = os.getenv("ANTHROPIC_LOCATION")

if not PROJECT_ID or not LOCATION:
    raise ValueError("ANTHROPIC_PROJECT_ID and ANTHROPIC_LOCATION must be set in the environment variables")

app = FastAPI()

class OpenAIToAnthropicTranslator:
    def __init__(self, project_id: str, location: str):
        self.client = AnthropicVertex(project_id=project_id, region=location)
        self.model = "claude-3-sonnet@20240229"
        self.vector_store = self.initialize_vector_store()
        self.index = self.create_index()
        self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def initialize_vector_store(self):
        dimension = 384  # Dimension of the HuggingFaceEmbedding
        return FaissVectorStore(faiss_index=faiss.IndexFlatL2(dimension))

    def create_index(self):
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        service_context = ServiceContext.from_defaults(embed_model=embed_model)

        # Create a storage context with the vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        # Create an empty index
        return VectorStoreIndex.from_documents(
            [],
            storage_context=storage_context,
            service_context=service_context
        )

    def add_to_memory(self, text: str):
        doc = Document(text=text)
        self.index.insert(doc)

    def query_memory(self, query: str, top_k: int = 5) -> List[str]:
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return [node.text for node in response.source_nodes[:top_k]]

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: Optional[Dict[str, float]] = None,
        user: str = "",
        files: Optional[List[UploadFile]] = None
    ) -> Dict[str, Any]:
        # Extract system message if present
        system_message = None
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                filtered_messages.append(message)

        # Query memory for relevant information
        last_user_message = next((m["content"] for m in reversed(filtered_messages) if m["role"] == "user"), None)
        if last_user_message:
            relevant_info = self.query_memory(last_user_message)
            context = "\n".join(relevant_info)
            system_message = f"{system_message}\n\nRelevant context:\n{context}" if system_message else f"Relevant context:\n{context}"

        anthropic_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": filtered_messages,
            "temperature": temperature,
            "top_p": top_p,
            "stop_sequences": stop or [],
            "stream": stream,
        }

        # Add system message if present
        if system_message:
            anthropic_params["system"] = system_message

        # Handle file uploads
        if files:
            file_contents = []
            for file in files:
                file_content = file.file.read()
                file_contents.append({"type": "image", "data": file_content})
            anthropic_params["files"] = file_contents

        logger.debug(f"Anthropic API request parameters: {anthropic_params}")

        try:
            response = self.client.messages.create(**anthropic_params)

            if not stream:
                # Add the assistant's response to memory
                self.add_to_memory(response.content[0].text)

            return response
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error calling Anthropic API: {str(e)}")

translator = OpenAIToAnthropicTranslator(PROJECT_ID, LOCATION)

class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = "gpt-3.5-turbo"
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = ""

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

def stream_chat_completion(request_data: dict, files: Optional[List[UploadFile]] = None):
    try:
        response = translator.create_chat_completion(**request_data, files=files)

        created_timestamp = int(time.time())
        message_id = f"chatcmpl-{created_timestamp}"
        full_response = ""

        for chunk in response:
            if isinstance(chunk, anthropic.types.RawContentBlockStartEvent):
                yield "data: " + json.dumps({
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": translator.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                            },
                            "finish_reason": None
                        }
                    ]
                }) + "\n\n"
            elif isinstance(chunk, anthropic.types.RawContentBlockDeltaEvent):
                full_response += chunk.delta.text
                yield "data: " + json.dumps({
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": translator.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk.delta.text,
                            },
                            "finish_reason": None
                        }
                    ]
                }) + "\n\n"
            elif isinstance(chunk, anthropic.types.RawContentBlockStopEvent):
                yield "data: " + json.dumps({
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created_timestamp,
                    "model": translator.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }) + "\n\n"

        yield "data: [DONE]\n\n"

        # Generate embedding for the full response
        embedding = translator.embed_model.get_text_embedding(full_response)

        # Add the complete response to memory
        translator.add_to_memory(full_response)

        logger.info(f"Generated embedding for streaming response: {embedding[:5]}...")  # Log first 5 elements of embedding

    except Exception as e:
        logger.error(f"Error in stream_chat_completion: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

async def parse_request(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    messages: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    top_p: Optional[float] = Form(None),
    n: Optional[int] = Form(None),
    stream: Optional[bool] = Form(None),
    stop: Optional[str] = Form(None),
    presence_penalty: Optional[float] = Form(None),
    frequency_penalty: Optional[float] = Form(None),
    logit_bias: Optional[str] = Form(None),
    user: Optional[str] = Form(None)
):
    if request.headers.get("Content-Type", "").startswith("application/json"):
        json_data = await request.json()
    else:
        json_data = {}
        if messages:
            json_data["messages"] = json.loads(messages)
        for field in ["model", "max_tokens", "temperature", "top_p", "n", "stream", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user"]:
            value = locals()[field]
            if value is not None:
                json_data[field] = value if field != "stop" and field != "logit_bias" else json.loads(value)

    return ChatCompletionRequest(**json_data), files

@app.post("/v1/chat/completions")
async def create_chat_completion(request_data: Tuple[ChatCompletionRequest, Optional[List[UploadFile]]] = Depends(parse_request)):
    request, files = request_data
    logger.info(f"Received request: {jsonable_encoder(request)}")

    try:
        if request.stream:
            return StreamingResponse(stream_chat_completion(request.dict(), files), media_type="text/event-stream")
        else:
            response = translator.create_chat_completion(**request.dict(), files=files)
            created_timestamp = int(time.time())
            return {
                "id": f"chatcmpl-{created_timestamp}",
                "object": "chat.completion",
                "created": created_timestamp,
                "model": translator.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response.content[0].text
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            }
    except Exception as e:
        logger.error(f"Error in create_chat_completion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server with PROJECT_ID: {PROJECT_ID}, LOCATION: {LOCATION}")
    uvicorn.run(app, host="0.0.0.0", port=8200)
