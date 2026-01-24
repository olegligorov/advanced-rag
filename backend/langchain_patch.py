"""
Monkey patch for langchain-community BaseBlobParser import issue.

This fixes the ImportError when using langchain-community 0.4.x with langchain-core 1.2.x
"""

import sys
from unittest.mock import MagicMock

# Create a mock BaseBlobParser before langchain_community tries to import it
if 'langchain_core.document_loaders' not in sys.modules:
    # Import the module first
    import langchain_core.document_loaders as loaders_module

    # Check if BaseBlobParser is missing
    if not hasattr(loaders_module, 'BaseBlobParser'):
        # Add the missing BaseBlobParser as a mock
        loaders_module.BaseBlobParser = type('BaseBlobParser', (), {})
        print("Applied langchain BaseBlobParser patch")
