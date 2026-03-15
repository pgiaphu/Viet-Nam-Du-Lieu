# src/sql_agent_system/tools/knowledge_tool.py

"""
Custom Knowledge Tool for CrewAI
Replacement for TextFileKnowledgeSource with full control
"""

import os
import requests
import hashlib
from typing import List, Dict, Optional
from pydantic import Field
from crewai.tools import BaseTool


# ============================================================================
# CONFIGURATION
# ============================================================================

OLLAMA_URL = os.getenv("EMBEDDINGS_OLLAMA_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDINGS_OLLAMA_MODEL_NAME")
CHUNK_SIZE = 400
TOP_K_RESULTS = 3
EMBEDDING_TIMEOUT = 30
EMBEDDING_RETRIES = 3


# ============================================================================
# VECTOR KNOWLEDGE BASE
# ============================================================================

class VectorKnowledgeBase:
    """Lightweight vector store for knowledge retrieval"""
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.embedding_model = embedding_model
        self.chunks: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []
        self.embedding_cache: Dict[str, List[float]] = {}  # ✅ Cache embeddings
        self.failed_chunks: List[int] = []  # ✅ Track failed embeddings
        
    def load_from_file(self, file_path: str) -> int:
        """Load and embed knowledge from text file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Knowledge file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Smart chunking by paragraphs
        raw_chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        
        # Split large chunks
        final_chunks = []
        for chunk in raw_chunks:
            if len(chunk) <= CHUNK_SIZE:
                final_chunks.append(chunk)
            else:
                sentences = chunk.split('. ')
                temp_chunk = []
                for sent in sentences:
                    temp_chunk.append(sent)
                    if len('. '.join(temp_chunk)) >= CHUNK_SIZE:
                        final_chunks.append('. '.join(temp_chunk).strip())
                        temp_chunk = []
                if temp_chunk:
                    final_chunks.append('. '.join(temp_chunk).strip())
        
        # Create embeddings with retry and error tracking
        successful_count = 0
        for i, chunk in enumerate(final_chunks):
            embedding = self._get_embedding_with_retry(chunk)
            if embedding:
                self.chunks.append(chunk)
                self.embeddings.append(embedding)
                self.metadata.append({'source': file_path, 'chunk_id': i})
                successful_count += 1
            else:
                self.failed_chunks.append(i)
                print(f"⚠️  Failed to embed chunk {i}: {chunk[:50]}...")
        
        if successful_count < len(final_chunks):
            print(f"⚠️  Warning: Only {successful_count}/{len(final_chunks)} chunks embedded successfully")
            print(f"   Failed chunks will be skipped in searches")
        
        return successful_count
    
    def _get_embedding_with_retry(self, text: str) -> Optional[List[float]]:
        """Get embedding with retry logic and caching"""
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Try multiple times with retry
        for attempt in range(EMBEDDING_RETRIES):
            try:
                embedding = self._get_embedding(text)
                if embedding:
                    # Cache successful result
                    self.embedding_cache[text_hash] = embedding
                    return embedding
            except Exception as e:
                if attempt < EMBEDDING_RETRIES - 1:
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⚠️  Embedding attempt {attempt + 1} failed: {str(e)[:100]}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        return None
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Search for relevant chunks with fallback to keyword matching"""
        query_embedding = self._get_embedding_with_retry(query)
        
        # If embedding fails, fall back to keyword matching
        if not query_embedding:
            print(f"⚠️  Embedding failed for query. Using keyword fallback...")
            return self._keyword_search(query, top_k)
        
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scores.append({
                'text': self.chunks[i],
                'score': similarity,
                'metadata': self.metadata[i]
            })
        
        results = sorted(scores, key=lambda x: x['score'], reverse=True)[:top_k]
        
        # If embedding search yields low scores, also try keyword search
        if results and results[0]['score'] < 0.5:
            print(f"⚠️  Low confidence in semantic search (score: {results[0]['score']:.2f}). Checking keyword matches...")
            keyword_results = self._keyword_search(query, top_k)
            if keyword_results:
                return keyword_results
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search when embeddings fail"""
        query_words = set(query.lower().split())
        
        scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            # Count matching words
            matches = len(query_words & chunk_words)
            if matches > 0:
                scores.append({
                    'text': chunk,
                    'score': matches / len(chunk_words),  # Relevance as ratio
                    'metadata': self.metadata[i]
                })
        
        return sorted(scores, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama with error reporting"""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=EMBEDDING_TIMEOUT
            )
            if response.status_code == 200:
                return response.json().get('embedding')
            else:
                raise Exception(f"Ollama returned status {response.status_code}: {response.text[:200]}")
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama timeout after {EMBEDDING_TIMEOUT}s - service may be slow")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {OLLAMA_URL} - service may be down")
        except Exception as e:
            raise Exception(f"Embedding error: {str(e)[:100]}")
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(a * a for a in vec2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


# ============================================================================
# CREWAI KNOWLEDGE TOOL
# ============================================================================

class ForecastKnowledgeTool(BaseTool):
    """
    Knowledge tool for forecast error metrics
    Replaces TextFileKnowledgeSource with better control
    """
    
    name: str = "search_forecast_knowledge"
    description: str = """Search internal knowledge base for forecast error metrics (MAPE, RMSE, MAE, WMAPE) and SQL formulas.
    
    Use this tool when you need:
    - SQL formulas for forecast metrics
    - Definitions of error metrics
    - Business context for forecast analysis
    - Who is responsible for each product category/line
    
    Input: Your question as a string
    Output: Relevant information from knowledge base with source citations
    
    IMPORTANT: Always use this tool before writing SQL queries for forecast metrics or category assignments."""
    
    kb: Optional[VectorKnowledgeBase] = Field(default=None)
    
    def _run(self, query: str) -> str:
        """Execute knowledge search with error handling"""
        if not self.kb:
            return "⚠️ Knowledge base not initialized"
        
        try:
            results = self.kb.search(query, top_k=TOP_K_RESULTS)
            
            if not results:
                return "ℹ️ No relevant information found in knowledge base for this query."
            
            # Format results with clear structure
            output = "📚 FORECAST KNOWLEDGE BASE:\n" + "="*70 + "\n\n"
            
            for i, result in enumerate(results, 1):
                output += f"[SOURCE {i}] (Relevance: {result['score']:.2f})\n"
                output += f"{result['text']}\n"
                output += "-"*70 + "\n\n"
            
            output += "💡 Use the information above when generating SQL queries.\n"
            return output
        except Exception as e:
            return f"⚠️ Knowledge search error: {str(e)[:200]}\n\nPlease try again or rephrase your question."


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_knowledge_tool(knowledge_file: str) -> ForecastKnowledgeTool:
    """
    Initialize knowledge tool from file
    
    Args:
        knowledge_file: Path to knowledge .txt file
        
    Returns:
        ForecastKnowledgeTool ready for CrewAI agents
        
    Example:
        tool = create_knowledge_tool("knowledge/forecast_metrics.txt")
        agent = Agent(..., tools=[tool])
    """
    print(f"📚 Loading knowledge from: {knowledge_file}")
    
    # Create and populate knowledge base
    kb = VectorKnowledgeBase(embedding_model=EMBEDDING_MODEL)
    num_chunks = kb.load_from_file(knowledge_file)
    
    print(f"✅ Knowledge base ready: {num_chunks} chunks indexed")
    
    # Create tool
    tool = ForecastKnowledgeTool(kb=kb)
    return tool