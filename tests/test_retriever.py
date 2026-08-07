from retriever import Retriever
from sentence_transformers import SentenceTransformer
import numpy as np

retriever = Retriever('data/test.txt')

def test_retriever_generates_text_upon_instantiation():
    assert retriever.text is not None
    assert type(retriever.text) == str

def test_retriever_generates_chunks_upon_instantiation():
    assert retriever.chunks is not None
    assert type(retriever.chunks) == list

def test_retriever_saves_embeddings_upon_instantiation():
    assert retriever.tuple is not None
    em_ch_tuples = list(retriever.tuple)
    assert type(em_ch_tuples[0]) == tuple
    assert type(em_ch_tuples[0][0]) == np.ndarray
    assert type(em_ch_tuples[0][1]) == str
    assert em_ch_tuples[0][1] == 'test text'

def test_retrieve_chunks_returns_list_of_top_3_chunks():
    prompt = 'test'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(prompt)
    result = retriever.retrieve_top_3_chunks(embeddings)
    assert type(result) == list
    assert len(result) == 3
    assert all(isinstance(item, str) for item in result)