import openparse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

basic_doc_path = "./Rohit_Ojha_Resume.pdf"
parser = openparse.DocumentParser()
parsed_basic_doc = parser.parse(basic_doc_path)


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


embeddings = []
texts = []

for node in parsed_basic_doc.nodes:
    text = node.text.strip()

    if text:
        embedding = model.encode(text)
        embeddings.append(embedding)
        texts.append(text)

embedding_matrix = np.array(embeddings).astype('float32')
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(embedding_matrix)


query_text = "automation tools like selenium"
query_embedding = model.encode([query_text]).astype('float32')

# Search top 3 similar embeddings
D, I = index.search(query_embedding, k=1)  # D = distance, I = indices

# I will contain indices of matching embeddings (node1, node2, etc.)

for idx in I[0]:
    print(f"- {texts[idx]}")









