#!/usr/bin/env python3
"""
Test chunking on documentation files to see how semantic chunking works
"""

import sys
from pathlib import Path
from rag.chunking import Chunker
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def main():
    # Allow file path as command line argument, default to entity-model.md
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    else:
        # Default to entity-model.md for demonstration
        DOCS_PATH = Path(__file__).parent.parent / "k8s_data" / "concepts" / "workloads" / "pods"
        file_path = DOCS_PATH / "pod-lifecycle.md"

    print("="*80)
    print(f"Testing Semantic Chunking on: {file_path.name}")
    print("="*80)

    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        print(f"\nUsage: python {Path(__file__).name} [path/to/file.md]")
        return

    print(f"\n✅ Loading file: {file_path}")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"📄 File size: {len(content)} characters")
    print(f"📝 Contains {content.count('```')} code block markers")
    print(f"📝 Contains {content.count('#')} header markers")

    # Create document
    from langchain_core.documents import Document
    doc = Document(page_content=content, metadata={"source": str(file_path)})

    # Test chunking
    print("\n" + "="*80)
    print("Running Semantic Chunker...")
    print("="*80)

    chunker = Chunker()
    chunks = chunker.create_chunks([doc])

    print(f"\n✅ Created {len(chunks)} chunks\n")

    # Analyze each chunk
    for i, chunk in enumerate(chunks, 1):
        chunk_content = chunk.page_content

        print(f"\n{'='*80}")
        print(f"CHUNK {i}/{len(chunks)} (length: {len(chunk_content)} chars)")
        print(f"{'='*80}")

        # Show the full content
        print(chunk_content)

        # Basic analysis
        has_code_blocks = "```" in chunk_content
        line_count = chunk_content.count('\n') + 1

        # Extract first header if present
        first_header = None
        for line in chunk_content.split('\n'):
            if line.strip().startswith('#'):
                first_header = line.strip()
                break

        print(f"\n📊 Chunk Stats:")
        print(f"  - Lines: {line_count}")
        print(f"  - Characters: {len(chunk_content)}")
        print(f"  - Contains code blocks: {'✅' if has_code_blocks else '❌'}")
        if first_header:
            print(f"  - First header: {first_header}")

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total_chars = sum(len(c.page_content) for c in chunks)
    avg_chunk_size = total_chars // len(chunks) if chunks else 0
    chunks_with_code = sum(1 for c in chunks if "```" in c.page_content)

    print(f"📊 Chunking Results:")
    print(f"  - Original file size: {len(content)} characters")
    print(f"  - Total chunks created: {len(chunks)}")
    print(f"  - Average chunk size: {avg_chunk_size} characters")
    print(f"  - Chunks with code blocks: {chunks_with_code}")
    print(f"\n✅ Chunking completed successfully!")

if __name__ == "__main__":
    main()
