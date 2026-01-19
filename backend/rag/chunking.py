import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
import hashlib

from config import EMBEDDING_MODEL, CHUNK_PERCENTILE

class SemanticChunker:
    """
    Semantic chunking implementation that splits documents based on topic shifts.

    Uses sentence embeddings and cosine similarity to detect semantic boundaries
    between sentences. When the distance between consecutive sentences exceeds
    a threshold, a new chunk is created.

    This approach is more intelligent than fixed-size chunking because it:
    - Preserves semantic coherence within chunks
    - Adapts to the natural structure of the text
    - Avoids splitting related content
    """

    def __init__(self, embedding_model=None):
        """
        Initialize SemanticChunker with an embedding model.

        Args:
            embedding_model (SentenceTransformer, optional): Pre-loaded SentenceTransformer model.
                If None, creates a new model using EMBEDDING_MODEL from config.
                Passing a pre-loaded model saves memory when sharing across components.
        """
        if embedding_model is not None:
            self.model = embedding_model
        else:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
    
    def create_semantic_chunks(self, docs, percentile_threshold=CHUNK_PERCENTILE):
        """
        Split documents into semantically coherent chunks based on topic shifts.

        Algorithm:
        1. Split each document into sentences
        2. Encode sentences using the embedding model
        3. Calculate cosine distance between consecutive sentence embeddings
        4. Use percentile-based threshold to identify semantic breakpoints
        5. Create new chunks at breakpoints where distance exceeds threshold

        Args:
            docs (list[Document]): List of LangChain Document objects to chunk.
            percentile_threshold (int, optional): Percentile value (0-100) for determining
                breakpoints. Higher values = fewer, larger chunks. Defaults to CHUNK_PERCENTILE (95).

        Returns:
            list[Document]: List of semantically chunked Document objects with original metadata.

        Example:
            If percentile_threshold=95, only the top 5% largest distances between
            sentences will trigger chunk splits, resulting in fewer but more coherent chunks.
        """
    
        all_chunks = []
        all_distances = []
        min_sentence_length = 10
        for doc in docs:
            text = doc.page_content
            metadata = doc.metadata
            
            sentences = re.split(r'(?<=[.?!])\s+|\n+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > min_sentence_length]
            
            if len(sentences) < 2:
                all_chunks.append(doc)
                continue
            
            embeddings = self.model.encode(sentences)
            distances = []
            
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                # use distance as 1 - similiarity 
                distances.append(1 - sim)
                
            all_distances.extend(distances)
            
            # TODO check this
            # use percentile to determine breakpoints, if distance exceeds this then split into new chunk
            breakpoint_threshold = np.percentile(distances, percentile_threshold)

            current_chunk_sentences = [sentences[0]]
            for i, distance in enumerate(distances):
                if distance > breakpoint_threshold:
                    all_chunks.append(Document(
                        page_content=" ".join(current_chunk_sentences),
                        metadata=metadata
                    ))
                    current_chunk_sentences = [sentences[i + 1]]
                else:
                    current_chunk_sentences.append(sentences[i + 1])
            
            # add the last chunk
            all_chunks.append(Document(
                page_content=" ".join(current_chunk_sentences),
                metadata=metadata
            ))
            
        print(f"Docs split into {len(all_chunks)} semantic segments.")
        # check if needed to enable these things for debugging 
        # return all_chunks, all_distances, breakpoint_threshold
        return all_chunks
