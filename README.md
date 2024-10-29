# Chatbot
## Introduction

This API allows users to interact with a chatbot that answers questions based on text content extracted from provided URLs. The chatbot utilizes conversational AI models and a document retrieval system to generate responses.

## Getting Started

To use the API, follow these steps:

1. Clone this repository:   https://github.com/DarshanG12/Chat_With_URL.git

2. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Run the Flask application:
```
python app.py
```


The API will be accessible at `http://localhost:5000/`.

## Usage

To interact with the chatbot, send a POST request to the `/chat` endpoint with a JSON object containing a URL and a question. Here's an example using cURL:
```
curl -X POST -H "Content-Type: application/json" -d '{"url": "https://example.com/article", "question": "What is the main topic of the article?"}' http://localhost:5000/chat
```

The API will respond with the chatbot's answer to the question.

## Documentation

For more information on how to use the API, refer to the doccumentation.

