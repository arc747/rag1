from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.evaluation import load_evaluator


# embeds = OllamaEmbeddings(model="llama2")
# vec = embeds.embed_query("an apple")
# print(len(vec))


def generate_embeddings():
    embedding_obj = OllamaEmbeddings(model="llama3")
    # embedding_vectors = embedding_obj.embed_query(query)
    return embedding_obj
    

def pairwise_evaluator(str1: str, str2: str, model=OllamaEmbeddings(model="llama3")):
    embedding_model = model
    evaluator =load_evaluator("pairwise_embedding_distance", embeddings=embedding_model)
    score = evaluator.evaluate_string_pairs(prediction=str1, prediction_b=str2)
    print("Comparing: ", str1, "&", str2, ": ", score["score"])


if __name__=="__main__":
    pairwise_evaluator("Apple", "Nokia N95")
    pairwise_evaluator("Apple", "Orange")
    pairwise_evaluator("Apple", "Nokia")
    pairwise_evaluator("Orange", "Nokia")
    pairwise_evaluator("Orange", "The Nederlands")
    pairwise_evaluator("Dutch", "The Nederlands")