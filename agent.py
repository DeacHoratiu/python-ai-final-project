import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, ConfigDict, Field, ValidationError



MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cool_db"


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description="Question about the selected document."
    )

class SearchResponse(BaseModel):
    context: str = Field(strict=True)


def rag_search(collection,model,file_name,question):
    request = SearchRequest(question=question)

    query_embeddings = model.encode(request.question, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embeddings],
        n_results=10,
        where={"source": file_name},
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_pieces = []

    for i in range(len(documents)):
        document = documents[i]
        metadata = metadatas[i]

        context_pieces.append(
            f"{document}\nSource: {metadata['source']}, page {metadata['page']}"
        )

    if context_pieces:
        context = "\n\n".join(context_pieces)
    else:
        context = "No relevant information found in the selected document"

    response = SearchResponse(context=context)

    return response.context


def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    model = SentenceTransformer(MODEL_NAME)

    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8000,
)

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    results = collection.get(include=["metadatas"])
    available_files = []

    for metadata in results["metadatas"]:
        file_name = metadata["source"]

        if file_name not in available_files:
            available_files.append(file_name)

    print("\nWelcome to Ancient Civilizations AI!")
    print("\nChoose a document:")
    print("\n".join(available_files))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": (
                    "Search the selected document for information needed to answer the user's question"
                ),
            "parameters": SearchRequest.model_json_schema(),
            }
        }
    ]

    while True:
        file_name = input("\nyou> Enter file name or 'quit' if you want to leave (dir for available docs): ").strip()

        if file_name.lower() == "dir":
            print("\n".join(available_files))
            continue
        if file_name.lower() == "quit":
            break

        if file_name not in available_files:
            print("Unknown file. Pick one of the following:")
            print("\n".join(available_files))
            continue

        question = input(f"you> What question do you have about {file_name}?\n").strip()

        if not question:
            print("Question cannot be empty")
            continue
        try:
            request = SearchRequest(question=question)
        except ValidationError:
            print("Invalid question. Please enter text")
            continue

        question = request.question

        if question.lower() == "quit":
            break

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in ancient civilizations. "
                    f"The selected document is {file_name}. "
                    "Use rag_search before answering. "
                    "Answer only from the context returned by the tool. "
                    "If the context is insufficient, say so. "
                    "Cite the source file and page number."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        response = client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL"),
            messages=messages,
            tools=tools,
            tool_choice="auto",
        ).choices[0].message

        messages.append(response.model_dump(exclude_none=True))

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "rag_search":
                    try:
                        tool_request = SearchRequest.model_validate_json(tool_call.function.arguments)

                        output = rag_search(
                            collection=collection,
                            model=model,
                            file_name=file_name,
                            question=tool_request.question,
                        )
                    except ValidationError:
                            output = "Invalid arguments for rag_search"
                else:
                    output = "Unknown tool"

                messages.append({
                    "role": "tool",
                    "tool_call_id":tool_call.id,
                    "content":output,
                })

            response = client.chat.completions.create(
                model=os.getenv("OPENROUTER_MODEL"),
                messages=messages,
            ).choices[0].message

        print(f"agent> {response.content}")

if __name__ == "__main__":
    main()
