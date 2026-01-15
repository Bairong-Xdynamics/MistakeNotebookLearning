"""
Prompt Builder for constructing system prompts from knowledge base entries.

Organizes guidance by subjects and formats into structured system prompts.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from .utils import count_tokens, truncate_text


class PromptBuilder:
    """
    Builds system prompts from knowledge base entries.
    
    Organizes guidance by subject and creates well-structured system prompts.
    """
    
    def __init__(
        self,
        max_total_tokens: Optional[int] = None,
        token_model: str = "gpt-3.5-turbo",
    ):
        """
        Initialize the prompt builder.
        
        Args:
            max_total_tokens: Maximum total tokens for the system prompt (None for no limit)
            token_model: Model name for token counting
        """
        self.max_total_tokens = max_total_tokens
        self.token_model = token_model
    
    def build_system_prompt(
        self,
        knowledge_entries: List[Dict[str, Any]],
        include_header: bool = True,
    ) -> str:
        """
        Build a structured system prompt from knowledge base entries.
        
        Args:
            knowledge_entries: List of knowledge base entries with 'subject' and 'guidance'
            include_header: Whether to include an introductory header
            
        Returns:
            Formatted system prompt string
        """
        if not knowledge_entries:
            return ""
        
        # Group entries by subject
        subject_groups = defaultdict(list)
        for entry in knowledge_entries:
            subject = entry.get("subject", "General")
            guidance = entry.get("guidance", "")
            if guidance:
                subject_groups[subject].append(guidance)
        
        # Build the prompt
        prompt_parts = []
        
        if include_header:
            prompt_parts.extend([   
        "The following mistake notes are not necessarily tied to the current question, but you may use them to deepen your analytical approach",
        "**IMPORTANT**: Before applying any guidance below, carefully evaluate:",
        "1. Does the current problem match the applicability conditions stated in the guidance?",
        "2. Is the problem type and context similar to the examples in the guidance?",
        "3. If the problem is fundamentally different (e.g., combinatorics vs modulo arithmetic, complex numbers vs number theory), do NOT force-fit the guidance.",
        "4. Only use guidance that is clearly relevant to the current problem structure and requirements.",
        ])
        # Add each subject section
        for subject, guidance_list in subject_groups.items():
            prompt_parts.append(f"## gudience subject: {subject} ##")
            prompt_parts.append("")
            
            # Combine guidance for this subject
            for guidance in guidance_list:
                prompt_parts.append(f"- {guidance}")
            
            prompt_parts.append("\n")
        
        prompt = "\n".join(prompt_parts).strip()+'\n\nBefore solving, review the attached guidance.State whether it is: "applicable", "partially applicable", or "irrelevant". Use only applicable parts when answering. '
        
        # Truncate if needed
        if self.max_total_tokens:
            current_tokens = count_tokens(prompt, self.token_model)
            if current_tokens > self.max_total_tokens:
                prompt = truncate_text(prompt, self.max_total_tokens, self.token_model)
        
        return prompt
    
    def build_simple_prompt(
        self,
        guidance_texts: List[str],
        separator: str = "\n\n",
    ) -> str:
        """
        Build a simple prompt from a list of guidance texts without subject grouping.
        
        Args:
            guidance_texts: List of guidance strings
            separator: Separator between guidance items
            
        Returns:
            Formatted system prompt string
        """
        if not guidance_texts:
            return ""
        
        prompt = separator.join(guidance_texts)
        
        # Truncate if needed
        if self.max_total_tokens:
            current_tokens = count_tokens(prompt, self.token_model)
            if current_tokens > self.max_total_tokens:
                prompt = truncate_text(prompt, self.max_total_tokens, self.token_model)
        
        return prompt
    
    def update_prompt_with_new_guidance(
        self,
        current_prompt: str,
        new_entries: List[Dict[str, Any]],
    ) -> str:
        """
        Update an existing prompt with new guidance entries.
        
        This extracts existing entries from the prompt, merges with new ones,
        and rebuilds the prompt.
        
        Args:
            current_prompt: Current system prompt
            new_entries: New knowledge base entries to add
            
        Returns:
            Updated system prompt
        """
        # For simplicity, we rebuild from the new entries
        # In a more sophisticated version, we could parse and merge
        return self.build_system_prompt(new_entries)
    
    def format_guidance_for_display(
        self,
        knowledge_entries: List[Dict[str, Any]],
    ) -> str:
        """
        Format knowledge entries for human-readable display.
        
        Args:
            knowledge_entries: List of knowledge base entries
            
        Returns:
            Formatted string for display
        """
        if not knowledge_entries:
            return "No guidance available."
        
        lines = ["# Current Guidance", ""]
        
        for i, entry in enumerate(knowledge_entries, 1):
            subject = entry.get("subject", "Unknown")
            guidance = entry.get("guidance", "")
            lines.append(f"{i}. **{subject}**")
            lines.append(f"   {guidance}")
            lines.append("")
        
        return "\n".join(lines)
    
    def extract_subjects_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract subject headings from a formatted prompt.
        
        Args:
            prompt: Formatted system prompt
            
        Returns:
            List of subject strings
        """
        subjects = []
        lines = prompt.split("\n")
        
        for line in lines:
            line = line.strip()
            if line.startswith("## "):
                subject = line[3:].strip()
                subjects.append(subject)
        
        return subjects
    
    def get_prompt_statistics(self, prompt: str) -> Dict[str, Any]:
        """
        Get statistics about a prompt.
        
        Args:
            prompt: System prompt
            
        Returns:
            Dictionary with statistics:
            - character_count: Number of characters
            - token_count: Number of tokens
            - line_count: Number of lines
            - subject_count: Number of subjects
        """
        char_count = len(prompt)
        token_count = count_tokens(prompt, self.token_model)
        line_count = len(prompt.split("\n"))
        subject_count = len(self.extract_subjects_from_prompt(prompt))
        
        return {
            "character_count": char_count,
            "token_count": token_count,
            "line_count": line_count,
            "subject_count": subject_count,
        }

