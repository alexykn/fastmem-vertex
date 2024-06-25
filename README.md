# fastmem-vertex

Understood, I've updated the README to focus more on the memory management and context retrieval aspects of the project:

## Memory-Augmented LLM Chat Assistant

This project is a FastAPI application that provides a chat interface for large language models (LLMs) like Anthropic or OpenAI. The key feature is its ability to augment the LLM's responses with relevant context retrieved from the conversation history using vector embeddings and the `llama-index` library.

### Features

- Retrieves relevant context from the conversation history using `llama-index` and vector embeddings
- Supports streaming and non-streaming responses from LLMs
- Supports file uploads to provide additional context
- Handles exceptions and provides detailed error messages
- Logging for debugging and monitoring
- Translates between OpenAI ChatGPT API and Anthropic API (currently Anthropic only)

### Future Goals

- Implement a persistent external database for long-term storage of conversation history
- Develop an automatic "memory pressure" based filtering mechanism to determine what gets appended to long-term storage and what stays only in memory
- Implement a score-based page replacement feature for efficient memory management
- Add support for other LLM providers (e.g., OpenAI, Cohere, etc.)

### Installation

1. Clone the repository:

```
git clone https://github.com/your-username/memory-augmented-llm-chat.git
```

2. Create a virtual environment and activate it:

```
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Set the environment variables for your LLM provider (e.g., Anthropic project ID and location):

```
export ANTHROPIC_PROJECT_ID=your-project-id
export ANTHROPIC_LOCATION=your-project-location
```

### Usage

To run the application, execute the following command:

```
uvicorn main:app --reload
```

This will start the FastAPI server at `http://localhost:8000`.

You can send requests to the `/v1/chat/completions` endpoint using tools like `curl` or a REST client like Postman.

#### Example Request

```
curl -X POST -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Hello, how are you?"}]}' http://localhost:8000/v1/chat/completions
```

This will send a request with a single user message and receive a response from the LLM, augmented with relevant context from the conversation history.

### Configuration

The application uses environment variables to configure the LLM provider settings (e.g., Anthropic project ID and location). You can set these variables in your shell or in a `.env` file in the project root directory.

```
ANTHROPIC_PROJECT_ID=your-project-id
ANTHROPIC_LOCATION=your-project-location
```

### Contributing

Contributions are welcome! Please follow the standard GitHub workflow:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them
4. Push your changes to your fork
5. Create a pull request

### License

See LICENSE file in main branch.
