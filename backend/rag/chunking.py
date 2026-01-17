import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
import hashlib

class SemanticChunker:
    def __init__(self, embedding_model=None):
        """
        Initialize SemanticChunker.

        Args:
            embedding_model: Optional SentenceTransformer model. If None, loads the default model.
        """
        # TODO set this as a config variable
        if embedding_model is not None:
            self.model = embedding_model
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def create_semantic_chunks(self, docs, percentile_threshold=95):
        """
        Takes a list of LangChain Document objects and splits them 
        into semantic chunks based on topic shifts.
        """
    
        all_chunks = []
        all_distances = []
        
        for doc in docs:
            text = doc.page_content
            metadata = doc.metadata
            
            sentences = re.split(r'(?<=[.?!])\s+|\n+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
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
