import time
from typing import List
import voyageai

try:
    from server.config import (
        VOYAGE_API_KEY,
        VOYAGE_MODEL_ID,
        VOYAGE_INPUT_TYPE,
        EMBEDDING_DIMENSIONS,
        MAX_TOKEN_LIMIT,
    )
except ImportError:
    from config import (
        VOYAGE_API_KEY,
        VOYAGE_MODEL_ID,
        VOYAGE_INPUT_TYPE,
        EMBEDDING_DIMENSIONS,
        MAX_TOKEN_LIMIT,
    )

class VoyageEmbeddingManager:
    _instance = None

    def __init__(self):
        print(f"[INIT] Initializing Voyage AI client...")
        print(f"[INIT] Target Model: '{VOYAGE_MODEL_ID}' (input_type: '{VOYAGE_INPUT_TYPE}')")
        print(f"[INIT] Token Limit: {MAX_TOKEN_LIMIT:,} tokens")
        start_t = time.time()
        
        self.model_id = VOYAGE_MODEL_ID
        self.input_type = VOYAGE_INPUT_TYPE
        self.dimensions = EMBEDDING_DIMENSIONS
        self.max_token_limit = MAX_TOKEN_LIMIT
        self.total_tokens_used = 0

        self.client = voyageai.Client(api_key=VOYAGE_API_KEY if VOYAGE_API_KEY else None)
        elapsed = time.time() - start_t
        print(f"[INIT] Voyage AI client initialized successfully in {elapsed:.2f}s.")

    @classmethod
    def get_instance(cls) -> "VoyageEmbeddingManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of texts using Voyage AI synchronous embedding API.
        Enforces MAX_TOKEN_LIMIT.
        """
        if not texts:
            return []

        if self.total_tokens_used >= self.max_token_limit:
            raise RuntimeError(
                f"Token limit reached! ({self.total_tokens_used:,} / {self.max_token_limit:,} tokens used). "
                f"Please update VOYAGE_API_KEY in the server configuration to continue."
            )

        clean_texts = [
            t if (isinstance(t, str) and t.strip()) else " "
            for t in texts
        ]

        try:
            result = self.client.embed(
                texts=clean_texts,
                model=self.model_id,
                input_type=self.input_type if self.input_type else None,
            )

            # Extract token count from Voyage AI response object
            batch_tokens = getattr(result, "total_tokens", 0)
            if not batch_tokens and hasattr(result, "usage"):
                usage = result.usage
                batch_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else getattr(usage, "total_tokens", 0)
            
            # Fallback estimation if token count is unavailable (approx 1 token per 4 chars)
            if not batch_tokens:
                batch_tokens = sum(max(1, len(t) // 4) for t in clean_texts)

            self.total_tokens_used += batch_tokens
            print(f"[INFO] Batch embedded: {len(clean_texts)} texts, {batch_tokens:,} tokens used. Cumulative: {self.total_tokens_used:,} / {self.max_token_limit:,} tokens.")

            if self.total_tokens_used >= self.max_token_limit:
                print(f"[WARNING] Maximum token limit of {self.max_token_limit:,} tokens has been reached!")

            return result.embeddings

        except Exception as e:
            if "Token limit reached" in str(e):
                raise
            print(f"[ERROR] Voyage AI API invocation failed ({type(e).__name__}): {e}")
            raise RuntimeError(f"VoyageAI ({type(e).__name__}): {e}") from e


