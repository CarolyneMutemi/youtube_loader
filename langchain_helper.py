from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = ChatOpenAI(temperature=0.7, streaming=True)
embeddings = OpenAIEmbeddings()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can answer questions about the given video transcript: {docs}. If a question is asked and you have an answer from your training data, you should provide it. If you don't have an answer, you should say that you don't have enough information to answer the question."),
    ("user", "{query}"),
])

def create_vectore_db(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url,
                                            language=["en", "en-US"],
                                            translation="en")
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    # print(transcript)
    # print("====================================")
    # print(docs)
    # loader = YoutubeLoader.from_youtube_url(
    #     video_url,
    #     transcript_format=TranscriptFormat.CHUNKS,
    #     chunk_size_seconds=30,
    # )
    # # print("\n\n".join(map(repr, loader.load())))
    # transcript = loader.load()

    # print(transcript)
    # print(docs)
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db: FAISS, query: str, num_results: int = 2):
    results = db.similarity_search(query, num_results)
    
    chain = prompt | llm
    response = chain.invoke({"query": query, "docs": results})
    print(response.content)

if __name__ == "__main__":
    vector_store = create_vectore_db("https://www.youtube.com/watch?v=tfU0JEZjcsg&t=906s")
    get_response_from_query(vector_store, "Explain the concept of S3 storage classes.")
