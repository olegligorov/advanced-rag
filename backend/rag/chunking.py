import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from config import EMBEDDING_MODEL, CHUNK_PERCENTILE

MAX_UNITS_PER_SECTION = 100
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000

class SemanticChunker:
    """
    RAG-Optimized Chunker for Technical Documentation.
    
    Improvements over standard semantic chunking:
    1. Respects Markdown Header hierarchy (Hard Splits).
    2. Treats Code Blocks as atomic units attached to preceding text (Soft Splits).
    3. Prevents splitting inside code snippets.
    """

    def __init__(self, embedding_model=None):
        if embedding_model is not None:
            self.model = embedding_model
        else:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

    def _split_by_markdown_headers(self, text: str, max_depth=3):
        """
        Performs a hard split on H1/H2/H3 (or deeper) headers to preserve high-level context.
        Returns a list of text sections.
        """
        pattern = rf'\n(?=#{{{1},{max_depth}}}\\s)'
        sections = re.split(pattern, text)
        return [s.strip() for s in sections if s.strip()]

    def _tokenize_text_with_code_preservation(self, text: str):
        """
        Splits text into 'Semantic Units'. 
        A Unit is either:
        - A single sentence
        - A sentence + the code block immediately following it
        """
        code_pattern = r'(```[\s\S]*?```|~~~[\s\S]*?~~~)'
        parts = re.split(code_pattern, text)
        
        semantic_units = []

        for part in parts:
            if not part:
                continue
                
            if re.match(code_pattern, part):
                if semantic_units:
                    semantic_units[-1] += f"\n\n{part}"
                else:
                    semantic_units.append(part)
            else:
                sentences = re.split(r'(?<=[.?!])\s+(?=[A-Z])', part)
                
                for s in sentences:
                    clean_s = s.strip()
                    if clean_s:
                        semantic_units.append(clean_s)

        return semantic_units

    def _process_section_with_size_constraints(self, semantic_units, metadata, percentile_threshold,
                                              min_size, max_size):
        """
        Process semantic units into chunks with size constraints.
        Returns a list of Document chunks.
        """
        chunks = []

        if len(semantic_units) == 1:
            chunks.append(Document(page_content=semantic_units[0], metadata=metadata))
            return chunks

        # Step 1: Embed Units
        embeddings = self.model.encode(semantic_units, show_progress_bar=False)

        # Step 2: Calculate Cosine Distances
        distances = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distances.append(1 - sim)

        # Step 3: Determine Threshold
        breakpoint_threshold = np.percentile(distances, percentile_threshold)

        # Step 4: Create Chunks with size constraints
        current_chunk_units = [semantic_units[0]]

        for i, distance in enumerate(distances):
            next_unit = semantic_units[i + 1]
            current_size = sum(len(u) for u in current_chunk_units)
            next_size = len(next_unit)

            should_split = distance > breakpoint_threshold or (current_size + next_size > max_size)

            if current_size < min_size and distance <= breakpoint_threshold * 1.5:
                should_split = False

            if should_split and current_size >= min_size:
                chunk_text = "\n".join(current_chunk_units)
                chunks.append(Document(page_content=chunk_text, metadata=metadata))
                current_chunk_units = [next_unit]
            else:
                current_chunk_units.append(next_unit)

        if current_chunk_units:
            chunk_text = "\n".join(current_chunk_units)
            # Merge with previous if too small
            if chunks and len(chunk_text) < min_size:
                last_chunk = chunks[-1]
                merged_content = last_chunk.page_content + "\n" + chunk_text
                chunks[-1] = Document(page_content=merged_content, metadata=metadata)
            else:
                chunks.append(Document(page_content=chunk_text, metadata=metadata))

        return chunks

    def create_semantic_chunks(self, docs, percentile_threshold=CHUNK_PERCENTILE,
                              min_chunk_size=MIN_CHUNK_SIZE, max_chunk_size=MAX_CHUNK_SIZE):
        final_chunks = []

        for doc in docs:
            # Step 1: Structural Split (Headers)
            sections = self._split_by_markdown_headers(doc.page_content)

            for section in sections:
                # Step 2: Create Semantic Units (Sentences + Glued Code)
                semantic_units = self._tokenize_text_with_code_preservation(section)

                if not semantic_units:
                    continue

                # Handle large sections by splitting deeper
                if len(semantic_units) > MAX_UNITS_PER_SECTION:
                    # Try splitting by deeper headers (H3-H5)
                    subsections = self._split_by_markdown_headers(section, max_depth=5)

                    if len(subsections) > 1:
                        # Successfully split into smaller subsections
                        for subsection in subsections:
                            sub_units = self._tokenize_text_with_code_preservation(subsection)
                            if sub_units:
                                chunks = self._process_section_with_size_constraints(
                                    sub_units, doc.metadata, percentile_threshold,
                                    min_chunk_size, max_chunk_size
                                )
                                final_chunks.extend(chunks)
                    else:
                        # No deeper headers found, process as-is but warn
                        print(f"Warning: Large section with {len(semantic_units)} units (no subsections found)")
                        chunks = self._process_section_with_size_constraints(
                            semantic_units, doc.metadata, percentile_threshold,
                            min_chunk_size, max_chunk_size
                        )
                        final_chunks.extend(chunks)
                else:
                    chunks = self._process_section_with_size_constraints(
                        semantic_units, doc.metadata, percentile_threshold,
                        min_chunk_size, max_chunk_size
                    )
                    final_chunks.extend(chunks)

        print(f"Processed {len(docs)} docs into {len(final_chunks)} semantic chunks.")
        return final_chunks