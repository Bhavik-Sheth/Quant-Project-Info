"""
LLM Client with Gemini and Groq Support

Unified interface with automatic failover between providers.
"""

from typing import Optional, Dict, Any
import logging
from abc import ABC, abstractmethod

from RAG_Mentor.config import Config

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate response from LLM"""
        pass


class GeminiClient(BaseLLMClient):
    """Google Gemini LLM client"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key
            model: Model name (default: gemini-pro)
        """
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model_name = model or Config.GEMINI_MODEL
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate response from Gemini.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens
            
        Returns:
            Generated text
        """
        try:
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise


class GroqClient(BaseLLMClient):
    """Groq LLM client"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name (default: llama3-70b-8192)
        """
        self.api_key = api_key or Config.GROQ_API_KEY
        self.model_name = model or Config.GROQ_MODEL
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")
        
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate response from Groq.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum response tokens
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise


class UnifiedLLMClient:
    """
    Unified LLM client with automatic failover.
    
    Attempts Gemini first, falls back to Groq on failure.
    """
    
    def __init__(self):
        """Initialize unified client with both providers"""
        self.primary_client: Optional[BaseLLMClient] = None
        self.fallback_client: Optional[BaseLLMClient] = None
        
        # Initialize Gemini if available
        if Config.GEMINI_API_KEY:
            try:
                self.primary_client = GeminiClient()
                logger.info("Primary LLM: Gemini")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")
        
        # Initialize Groq if available
        if Config.GROQ_API_KEY:
            try:
                if self.primary_client is None:
                    self.primary_client = GroqClient()
                    logger.info("Primary LLM: Groq")
                else:
                    self.fallback_client = GroqClient()
                    logger.info("Fallback LLM: Groq")
            except Exception as e:
                logger.warning(f"Could not initialize Groq: {e}")
        
        if self.primary_client is None:
            raise ValueError("No LLM provider available. Set GEMINI_API_KEY or GROQ_API_KEY")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        retry: bool = True
    ) -> str:
        """
        Generate response with automatic failover.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            retry: Whether to retry with fallback on failure
            
        Returns:
            Generated text
        """
        # Try primary client
        try:
            return self.primary_client.generate(prompt, temperature, max_tokens)
        except Exception as e:
            logger.error(f"Primary LLM failed: {e}")
            
            # Try fallback if available
            if retry and self.fallback_client:
                logger.info("Attempting fallback LLM...")
                try:
                    return self.fallback_client.generate(prompt, temperature, max_tokens)
                except Exception as fallback_error:
                    logger.error(f"Fallback LLM also failed: {fallback_error}")
                    raise
            else:
                raise
    
    def get_active_provider(self) -> str:
        """
        Get name of currently active provider.
        
        Returns:
            Provider name
        """
        if isinstance(self.primary_client, GeminiClient):
            return "Gemini"
        elif isinstance(self.primary_client, GroqClient):
            return "Groq"
        else:
            return "Unknown"


# Global singleton instance
_llm_client: Optional[UnifiedLLMClient] = None


def get_llm_client() -> UnifiedLLMClient:
    """
    Get or create the global LLM client instance.
    
    Returns:
        UnifiedLLMClient instance
    """
    global _llm_client
    
    if _llm_client is None:
        _llm_client = UnifiedLLMClient()
    
    return _llm_client
