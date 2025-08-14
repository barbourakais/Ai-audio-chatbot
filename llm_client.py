"""
Ollama LLM client for the AI Audio Agent
"""

import requests
import json
import time
from typing import Optional, Dict, Any, List
from config import get_config
from utils import colored_print, setup_logging

logger = setup_logging()

class OllamaClient:
    """Client for interacting with Ollama LLM API"""
    
    def __init__(self):
        self.config = get_config("ollama")
        self.base_url = self.config["base_url"]
        self.model = self.config["model"]
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Cache the system prompt to avoid reloading
        self._cached_system_prompt = None
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                
                if self.model in model_names:
                    colored_print(f"✓ Connected to Ollama with model: {self.model}", "green")
                    return True
                else:
                    colored_print(f"⚠ Model '{self.model}' not found. Available models: {model_names}", "yellow")
                    colored_print("Please install the model with: ollama pull " + self.model, "yellow")
                    return False
            else:
                colored_print(f"✗ Failed to connect to Ollama: {response.status_code}", "red")
                return False
                
        except requests.exceptions.ConnectionError:
            colored_print("✗ Cannot connect to Ollama. Is it running?", "red")
            colored_print("Start Ollama with: ollama serve", "yellow")
            return False
        except Exception as e:
            colored_print(f"✗ Error testing Ollama connection: {e}", "red")
            return False
    
    def generate_response(self, prompt: str, context: str = "", 
                         system_prompt: Optional[str] = None) -> str:
        """Generate a response using the Ollama model"""
        try:
            # Prepare the full prompt with context
            if context:
                full_prompt = f"{context}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"
            
            # Use system prompt from config if not provided
            if system_prompt is None:
                if self.config["system_prompt"] is None:
                    # Use cached system prompt or load it
                    if self._cached_system_prompt is None:
                        from config import get_system_prompt
                        self._cached_system_prompt = get_system_prompt()
                    system_prompt = self._cached_system_prompt
                else:
                    system_prompt = self.config["system_prompt"]
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"],
                    "repeat_penalty": self.config["repeat_penalty"],
                    "num_predict": self.config["max_tokens"],
                }
            }
            
            colored_print("🤔 Thinking...", "cyan")
            
            # Make request to Ollama
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                if generated_text:
                    # Clean up the response
                    generated_text = self._clean_response(generated_text)
                    
                    # Limit response length for speech
                    if len(generated_text) > 400:  # Increased to 400 characters
                        sentences = generated_text.split('.')
                        if len(sentences) > 3:
                            generated_text = '. '.join(sentences[:3]).strip() + '.'
                        else:
                            generated_text = generated_text[:400].strip()
                    
                    colored_print(f"✓ Generated response ({len(generated_text)} chars)", "green")
                    return generated_text
                else:
                    colored_print("⚠ No response generated", "yellow")
                    return "I'm sorry, I couldn't generate a response. Please try again."
            
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f" - {error_data.get('error', 'Unknown error')}"
                except:
                    pass
                
                colored_print(f"✗ {error_msg}", "red")
                return "I'm sorry, there was an error generating the response. Please try again."
        
        except requests.exceptions.Timeout:
            colored_print("✗ Request timed out", "red")
            return "I'm sorry, the request timed out. Please try again."
        
        except requests.exceptions.ConnectionError:
            colored_print("✗ Connection lost to Ollama", "red")
            return "I'm sorry, I lost connection to the language model. Please check if Ollama is running."
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            colored_print(f"✗ Error: {e}", "red")
            return "I'm sorry, an unexpected error occurred. Please try again."
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Remove common prefixes that might be added by the model
        prefixes_to_remove = [
            "Assistant:",
            "AI:",
            "Bot:",
            "Response:",
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Remove any trailing incomplete sentences or partial lists
        if response.endswith("...") or response.endswith(".."):
            response = response.rstrip(".")
        
        # Remove incomplete numbered lists (e.g., "1." at the end)
        if response.rstrip().endswith("1.") or response.rstrip().endswith("2.") or response.rstrip().endswith("3."):
            # Find the last complete sentence
            sentences = response.split('.')
            if len(sentences) > 1:
                response = '. '.join(sentences[:-1]).strip() + '.'
        
        # Ensure the response ends with proper punctuation
        if response and not response.endswith((".", "!", "?")):
            response += "."
        
        return response.strip()
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            # Check if the model exists
            available_models = self.get_available_models()
            if model_name not in available_models:
                colored_print(f"Model '{model_name}' not found. Available: {available_models}", "red")
                return False
            
            # Test the model with a simple request
            test_payload = {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_predict": 10,
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                self.model = model_name
                colored_print(f"✓ Switched to model: {model_name}", "green")
                return True
            else:
                colored_print(f"✗ Failed to switch to model: {model_name}", "red")
                return False
                
        except Exception as e:
            colored_print(f"✗ Error switching model: {e}", "red")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = self.session.get(f"{self.base_url}/api/show", 
                                      params={"name": self.model})
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Perform a health check on the Ollama service"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_with_streaming(self, prompt: str, context: str = "", 
                               callback=None) -> str:
        """Generate response with streaming (for future use)"""
        try:
            if context:
                full_prompt = f"{context}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = f"User: {prompt}\nAssistant:"
            
            # Get dynamic system prompt
            if self.config["system_prompt"] is None:
                from config import get_system_prompt
                system_prompt = get_system_prompt()
            else:
                system_prompt = self.config["system_prompt"]
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "system": system_prompt,
                "stream": True,
                "options": {
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"],
                    "repeat_penalty": self.config["repeat_penalty"],
                    "num_predict": self.config["max_tokens"],
                }
            }
            
            full_response = ""
            
            with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            ) as response:
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            chunk = data.get("response", "")
                            full_response += chunk
                            
                            if callback:
                                callback(chunk)
                                
                        except json.JSONDecodeError:
                            continue
            
            return self._clean_response(full_response)
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            return "I'm sorry, an error occurred during response generation." 