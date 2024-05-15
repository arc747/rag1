import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from embeddings import generate_embeddings


CHROMA_PATH="chroma"

PROMPT_TEMPLATE = """
Answer the question based on the given context:
Context: {context}
\n\n
Question: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="the query text")
    args = parser.parse_args()
    query = args.query

    vdb = Chroma(persist_directory=CHROMA_PATH, embedding_function=generate_embeddings())
    context_search_results = vdb.similarity_search_with_relevance_scores(query, k=3)
    #  len(context_search)==0 or context_search[0][1]

    context = [doc.page_content for doc, score in context_search_results]
    sources = [doc.metadata for doc, score in context_search_results]
    # print(context)

    llm = Ollama(model="llama3")
    prompt_obj = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_obj.format(context=context, question=query)

    llm_response = llm.invoke(prompt)

    print(f"Response: {llm_response}\n\n Source: {sources}")


if __name__=="__main__":
    main()