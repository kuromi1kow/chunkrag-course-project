"""Protocol-frozen ChunkRAG main-study implementation.

Motivated by Immutable Specification Sections 1--33.  This package intentionally does
not import the legacy ``chunkrag.pipeline`` execution path.
"""

from .protocol import PROTOCOL_ID, PROTOCOL_SHA256, ProtocolError, load_protocol_config

__all__ = ["PROTOCOL_ID", "PROTOCOL_SHA256", "ProtocolError", "load_protocol_config"]
