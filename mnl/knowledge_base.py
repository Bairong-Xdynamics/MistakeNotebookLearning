"""
Knowledge Base for storing and retrieving guidance using RAG.

Uses embedding-based similarity search for efficient retrieval and
strong model for guidance merging and subject consolidation.
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
from .utils import load_jsonl, save_jsonl
from .llm_client import LLMClient


class KnowledgeBase:
    """
    RAG-based knowledge base for storing and retrieving guidance.
    
    Storage format: JSONL with {subject: str, guidance: str, embedding: List[float]}
    """
    
    def __init__(
        self,
        storage_path: str,
        llm_client: LLMClient,
        max_guidance_length: int = 1024,
        guidance_merge_prompt_template: Optional[str] = None,
    ):
        """
        Initialize the knowledge base.
        
        Args:
            storage_path: Path to JSONL storage file
            llm_client: LLM client for embeddings and merging
            max_guidance_length: Maximum character length for guidance
            guidance_merge_prompt_template: Custom template for merging guidance
        """
        self.storage_path = storage_path
        self.llm_client = llm_client
        self.max_guidance_length = max_guidance_length
        self.guidance_merge_prompt_template = guidance_merge_prompt_template or self._get_default_merge_prompt_template()
        
        # Load existing knowledge base
        self.entries = self._load_entries()
    
    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load entries from storage file."""
        if self.storage_path and os.path.exists(self.storage_path):
            return load_jsonl(self.storage_path)
        return []
    
    def _get_default_merge_prompt_template(self) -> str:
        """Get the default prompt template for merging guidance."""
        return (
            "You are synthesizing guidance for the subject: {subject}\n\n"
            "Existing guidance from related subjects in the knowledge base:\n"
            "{existing_guidance}\n\n"
            "New guidance to incorporate:\n{new_guidance}\n\n"
            "Please merge these guidance points into a single, coherent guidance text for '{subject}'.\n"
            "Consider insights from related subjects and adapt them to the current context.\n"
            "The output should be concise, clear, and no longer than {max_length} characters.\n"
            "Focus on the most important and actionable advice.\n\n"
            "Merged guidance:"
        )
    
    def _save_entries(self) -> None:
        """Save entries to storage file."""
        save_jsonl(self.entries, self.storage_path, append=False)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant guidance entries based on query similarity.
        
        Args:
            query: Query text (e.g., subject or question)
            top_k: Maximum number of entries to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant entries with similarity scores
        """
        if not self.entries:
            return []
        
        # Get query embedding
        query_embedding = self.llm_client.get_embedding(query)
        query_emb = np.array(query_embedding)
        
        # Filter entries with embeddings
        valid_entries = [entry for entry in self.entries if "embedding" in entry]
        if not valid_entries:
            return []
        
        # Vectorized similarity calculation using NumPy
        embeddings_matrix = np.array([entry["embedding"] for entry in valid_entries])
        
        # Calculate cosine similarities in batch
        # cos_sim = dot(A, B) / (norm(A) * norm(B))
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []
        
        # Compute dot products
        dot_products = np.dot(embeddings_matrix, query_emb)
        
        # Compute norms of all embeddings
        embedding_norms = np.linalg.norm(embeddings_matrix, axis=1)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            similarities_array = dot_products / (embedding_norms * query_norm)
            similarities_array = np.nan_to_num(
                similarities_array, nan=0.0, posinf=0.0, neginf=0.0
            )
        
        # Filter by threshold and create result list
        valid_indices = np.where(similarities_array >= threshold)[0]
        
        similarities = [
            {
                "entry": valid_entries[idx],
                "similarity": float(similarities_array[idx]),
            }
            for idx in valid_indices
        ]
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k
        return similarities[:top_k]
    
    def retrieve_by_subject(
        self,
        subject: str,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve guidance entries for a specific subject.
        
        Args:
            subject: Subject to search for
            top_k: Maximum number of entries to retrieve
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant entries with similarity scores
        """
        return self.retrieve(query=subject, top_k=top_k, threshold=threshold)
    
    def merge_guidance(
        self,
        retrieved_guidance: List[str],
        new_guidance: str,
        subject: str,
        related_subjects: Optional[List[str]] = None,
    ) -> str:
        """
        Merge retrieved guidance with new guidance using strong model.
        
        Args:
            retrieved_guidance: List of retrieved guidance texts
            new_guidance: New guidance to merge
            subject: Subject context for merging
            related_subjects: Optional list of subjects corresponding to retrieved guidance
            
        Returns:
            Merged and truncated guidance
        """
        if not retrieved_guidance:
            # No existing guidance, just truncate new guidance
            return new_guidance[:self.max_guidance_length]
        
        # Format existing guidance with related subjects if provided
        existing_guidance_parts = []
        for i, guidance in enumerate(retrieved_guidance, 1):
            if related_subjects and i <= len(related_subjects):
                existing_guidance_parts.append(f"{i}. [From: {related_subjects[i-1]}] {guidance}")
            else:
                existing_guidance_parts.append(f"{i}. {guidance}")
        
        existing_guidance_text = "\n".join(existing_guidance_parts)
        
        # Use the template to build the prompt
        prompt = self.guidance_merge_prompt_template.format(
            subject=subject,
            existing_guidance=existing_guidance_text,
            new_guidance=new_guidance,
            max_length=self.max_guidance_length
        )
        
        merged = self.llm_client.generate_with_tuner_model(
            prompt=prompt,
            temperature=0.3,
        )
        
        # Handle case where merging failed
        if merged is None:
            # Fallback: return the first retrieved guidance or new guidance
            if retrieved_guidance:
                merged = retrieved_guidance[0]
            else:
                merged = new_guidance
        
        # Truncate to max length
        return merged[:self.max_guidance_length]
    
    def merge_subjects(self, subjects: List[str]) -> str:
        """
        Merge similar subjects into a single consolidated subject.
        
        Args:
            subjects: List of subject strings
            
        Returns:
            Consolidated subject name
        """
        if len(subjects) <= 1:
            return subjects[0] if subjects else "General"
        
        prompt = (
            f"The following subjects are related:\n"
            f"{', '.join(subjects)}\n\n"
            f"Provide a single, concise subject name (2-5 words) that best represents all of these. "
            f"Only output the subject name, nothing else."
        )
        
        merged = self.llm_client.generate_with_tuner_model(
            prompt=prompt,
            temperature=0.3,
        )
        
        # Handle case where merging failed
        if merged is None:
            # Fallback: return the first subject
            return subjects[0][:50]
        
        # Clean up the response
        merged = merged.strip().strip('"').strip("'")
        return merged[:50]  # Limit subject name length
    
    def add_entry(
        self,
        subject: str,
        guidance: str,
        overwrite_if_exists: bool = False,
    ) -> None:
        """
        Add a new entry to the knowledge base.
        
        Args:
            subject: Subject label
            guidance: Guidance text
            overwrite_if_exists: If True, replace existing entry with same subject
        """
        # Truncate guidance if needed
        guidance = guidance[:self.max_guidance_length]
        
        # Get embedding for the subject
        embedding = self.llm_client.get_embedding(subject)
        
        entry = {
            "subject": subject,
            "guidance": guidance,
            "embedding": embedding,
        }
        
        # Check if subject already exists
        if overwrite_if_exists:
            self.entries = [e for e in self.entries if e.get("subject") != subject]
        self.entries.append(entry)
    
    def update_entry(
        self,
        subject: str,
        new_guidance: str,
    ) -> None:
        """
        Update an existing entry or create if not exists.
        
        Args:
            subject: Subject to update
            new_guidance: New guidance text
        """
        self.add_entry(subject, new_guidance, overwrite_if_exists=False)
    
    def get_all_subjects(self) -> List[str]:
        """
        Get all unique subjects in the knowledge base.
        
        Returns:
            List of subject strings
        """
        return [entry["subject"] for entry in self.entries]
    
    def get_entry_by_subject(self, subject: str) -> Optional[Dict[str, Any]]:
        """
        Get entry by exact subject match.
        
        Args:
            subject: Subject to search for
            
        Returns:
            Entry dict or None if not found
        """
        for entry in self.entries:
            if entry.get("subject") == subject:
                return entry
        return None
    
    def clear(self) -> None:
        """Clear all entries from the knowledge base."""
        self.entries = []
    
    def export_to_text(self) -> str:
        """
        Export knowledge base to human-readable text format.
        
        Returns:
            Formatted text representation
        """
        lines = ["# Knowledge Base", ""]
        
        for i, entry in enumerate(self.entries, 1):
            lines.append(f"## {i}. {entry['subject']}")
            lines.append(f"{entry['guidance']}")
            lines.append("")
        
        return "\n".join(lines)

