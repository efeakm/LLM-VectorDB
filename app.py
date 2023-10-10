import os

# Import chainlit for interactive scripting
import chainlit as cl
from chainlit.types import AskFileResponse

# Import document loaders for handling different types of documents (e.g. PDF, Text)
from langchain.document_loaders import PyPDFLoader, TextLoader
# Import text splitter for splitting documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Import embeddings for representation of text in vector form
from langchain.embeddings import OpenAIEmbeddings

# Additional imports related to the functionality but not used in this snippet
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI



# Initialize
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI-API-KEY"])

welcome_message = """
Welcome to the Document QA demo!
1. Upload a PDF document.
2. Ask a question.
"""

# Function to process an uploaded file
def process_file(file: AskFileResponse):
    # Import tempfile for creating temporary files
    import tempfile

    # Determine the type of the file and select the appropriate loader
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

    # Create a temporary file, write the content of the uploaded file into it, and process using the selected loader
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.content)
        loader = Loader(temp_file.name)

        # Load the document from the temporary file and split it into smaller chunks using the text splitter
        documents = loader.load()
        docs = text_splitter.split_documents(documents)

        # Add metadata to each chunk to identify its source
        for i, doc in enumerate(docs):
            doc.metadata['source'] = f'source_{i}'

        return docs


# Creates a Chroma vector store from a list of documents
def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set('docs', docs)

    # Creates the vector store
    docsearch = Chroma.from_documents(docs, embeddings)
    return docsearch


# On chat start, this decorator is triggered
@cl.on_chat_start
async def start():
    # Send an initial message to the user
    await cl.Message(content='You can now chat with your PDFs').send()

    # Initialize files to None. This will be used to hold the uploaded files
    files = None
    # Keep prompting the user to upload a file until they do so
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,  # Display the welcome message to the user
            accept=['text/plain', 'application/pdf'],  # Allowed file types for upload
            max_size_mb=20,  # Max file size of 20MB
            timeout=180,  # Wait for 180 seconds before timing out
        ).send()

    # Take the first file from the uploaded files (assuming only one file is uploaded)
    file = files[0]

    # Inform the user that the file is being processed
    await cl.Message(f'Processing {file.name}...').send()
    # Process the uploaded file asynchronously
    docsearch = await cl.make_async(get_docsearch)(file)

    # Set up the QA chain with the processed document to retrieve answers from
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0.0, streaming=True, openai_api_key=os.environ["OPENAI-API-KEY"]),  # Use OpenAI for QA with no randomness (temperature=0.0)
        chain_type='stuff',  # Type of the chain, specific value 'stuff' is used here but its significance is unclear without further context
        retriever=docsearch.as_retriever(max_tokens_limit=4097),  # Convert docsearch to a retriever with a token limit
    )
    # Store the created chain in the user's session for later use
    cl.user_session.set('chain', chain)

    # Notify the user that the file processing is complete and they can start asking questions
    await cl.Message(f"`{file.name}` processed. You can now ask questions!").send()


# Decorator to trigger this function on every received message
@cl.on_message
async def main(message):
    # Retrieve the previously stored QA chain from the user session
    chain = cl.user_session.get('chain')
    
    # Initialize an asynchronous callback handler for the chain, with specific tokens to recognize final answers
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=['FINAL', 'ANSWER']
    )
    cb.answer_reached = True  # Indicates that an answer has been reached

    # Invoke the QA chain with the user message and provided callback
    res = await chain.acall(message, callbacks=[cb])
    answer = res['answer']  # Extract the answer from the chain's response
    sources = res["sources"].strip()  # Extract the sources (if any) from the chain's response
    source_elements = []  # List to hold referenced sources

    # Retrieve the processed documents stored in the user session
    docs = cl.user_session.get('docs')
    metadatas = [doc.metadata for doc in docs]  # Extract metadata from each document
    all_sources = [m['source'] for m in metadatas]  # List of all sources

    # If there are sources in the chain's response
    if sources:
        found_sources = []

        # Process each source and add it to the message
        for source in sources.split(','):
            source_name = source.strip().replace('.', '')
            
            # Try to find the index of the source in all sources
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue  # If not found, continue to the next source

            text = docs[index].page_content  # Extract the content of the source document
            found_sources.append(source_name)
            # Create a text element with the content and add it to the list
            source_elements.append(cl.Text(content=text, name=source_name))

        # If there are any found sources, append them to the answer
        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"  # If no sources were found, add this to the answer

    # If a final answer was streamed, update the stream with the source elements
    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        # If no answer was streamed, send a new message with the answer and source elements
        await cl.Message(content=answer, elements=source_elements).send()




