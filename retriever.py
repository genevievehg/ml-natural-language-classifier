from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class Retriever:
    def __init__(self, doc_path):
        self.file_path = doc_path
        self.text = None
        self.chunks = None
        self.tuple = None
        self.read_doc()
        self.generate_chunks(self.text)
        self.generate_embeddings(self.chunks)

    def read_doc(self):
        with open(self.file_path) as file:
            text = file.read()
            self.text = text

    def generate_chunks(self, text):
        chunks = text.split('\n')
        self.chunks = chunks

    def generate_embeddings(self, chunks):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(chunks)
        tuple = zip(embeddings, chunks)
        self.tuple = tuple
        with open('data/models/embeddings.pkl', 'wb') as f:
            pickle.dump(tuple, f)

    def retrieve_top_3_chunks(self, prompt_embedding):
        with open('data/models/embeddings.pkl', 'rb') as f:
            existing_embeddings = pickle.load(f)
        cosine_similarities = []
        for em_ch_tuple in existing_embeddings:
            embedding = em_ch_tuple[0]
            chunk = em_ch_tuple[1]
            similarity = cosine_similarity([embedding], [prompt_embedding])[0][0]
            cosine_similarities.append((similarity, chunk))
        top_3 = sorted(cosine_similarities, key=lambda x: x[0], reverse=True)[:3]
        chunks = []
        for sim_ch_tuple in top_3:
            chunks.append(sim_ch_tuple[1])
        return chunks
