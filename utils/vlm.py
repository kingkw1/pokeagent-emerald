from io import BytesIO
from PIL import Image
import os
import base64
import random
import time
import logging
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, Optional
import numpy as np

# CRITICAL FIX: Unset GCP credentials BEFORE any Google imports
# This prevents the ALTS authentication error that causes agent stalls
# Must happen before google.generativeai is imported anywhere in the module
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    print("⚠️ [VLM] Unsetting GOOGLE_APPLICATION_CREDENTIALS to prevent ALTS authentication conflict")
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

# Set up module logging
logger = logging.getLogger(__name__)

# Import LLM logger
from utils.llm_logger import log_llm_interaction, log_llm_error

# Define the retry decorator with exponential backoff
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (Exception,),
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                # Increase the delay with exponential factor and random jitter
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(delay)
            except Exception as e:
                raise e
    return wrapper

class VLMBackend(ABC):
    """Abstract base class for VLM backends"""
    
    @abstractmethod
    def get_query(self, img: Union[Image.Image, np.ndarray, str], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt (accepts PIL Image, numpy array, or file path)"""
        pass
    
    @abstractmethod
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt"""
        pass

class OpenAIBackend(VLMBackend):
    """OpenAI API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            import openai
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: OpenAI API key is missing! Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.errors = (openai.RateLimitError,)
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using OpenAI API"""
        start_time = time.time()
        
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        
        try:
            response = self._call_completion(messages)
            result = response.choices[0].message.content
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": True}
            )
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using OpenAI API"""
        start_time = time.time()
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        try:
            response = self._call_completion(messages)
            result = response.choices[0].message.content
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage'):
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "openai", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "openai"}
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type=f"openai_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": "openai", "duration": duration, "has_image": False}
            )
            logger.error(f"OpenAI API error: {e}")
            raise

class OpenRouterBackend(VLMBackend):
    """OpenRouter API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: OpenRouter API key is missing! Set OPENROUTER_API_KEY environment variable.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using OpenRouter API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using OpenRouter API"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OPENROUTER VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result

class LocalHuggingFaceBackend(VLMBackend):
    """Local HuggingFace VLM backend with GPU support and optimization"""
    
    def _detect_model_type(self, model_name: str) -> str:
        """
        Detect the model type from name patterns or config file.
        
        Supports all trained models:
        - models/perception_v0.1/final_checkpoint (phi-3-vision)  
        - models/perception_v0.2_qwen* (qwen2-vl)
        - Hugging Face model names (qwen2-vl, phi-3, etc.)
        
        Returns:
            str: Model type ("qwen2_vl", "phi3_v", or "unknown")
        """
        # Direct name pattern matching for known models
        model_lower = model_name.lower()
        
        # Check for Qwen2-VL patterns
        if any(pattern in model_lower for pattern in [
            "qwen2-vl", "qwen2_vl", "perception_v0.2_qwen"
        ]):
            return "qwen2_vl"
            
        # Check for Phi-3 Vision patterns  
        if any(pattern in model_lower for pattern in [
            "phi-3-vision", "phi3-vision", "phi-3.5-vision", "perception_v0.1"
        ]):
            return "phi3_v"
        
        # For local checkpoint paths, try to read the config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config_type = getattr(config, 'model_type', '').lower()
            
            if config_type == "qwen2_vl":
                return "qwen2_vl"
            elif config_type == "phi3_v":
                return "phi3_v"
            else:
                logger.warning(f"Unknown model type from config: {config_type}")
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Could not read config for model type detection: {e}")
            
        # Default fallback - assume Phi-3 for backward compatibility
        logger.warning(f"Could not detect model type for {model_name}, defaulting to phi3_v")
        return "phi3_v"
    
    def __init__(self, model_name: str, device: str = "auto", load_in_4bit: bool = False, **kwargs):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
            from PIL import Image
        except ImportError as e:
            raise ImportError(f"Required packages not found. Install with: pip install torch transformers bitsandbytes accelerate. Error: {e}")
        
        self.model_name = model_name
        
        # Smart device selection - prefer GPU if available
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            self.device = device
            
        self.torch = torch
        
        logger.info(f"Loading local VLM model: {model_name}")
        
        # Configure quantization if requested
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization with bitsandbytes")
        
        # Load processor and model
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # Detect model type from name or config - supports all trained models
            self.model_type = self._detect_model_type(model_name)
            logger.info(f"Detected model type: {self.model_type}")
            
            if self.model_type == "qwen2_vl":
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device if device != "auto" else "auto",
                    torch_dtype=torch.float16 if not load_in_4bit else "auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map=device if device != "auto" else "auto",
                    torch_dtype=torch.float16 if not load_in_4bit else "auto",
                    trust_remote_code=True
                )
            
            if device != "auto" and not load_in_4bit:
                self.model = self.model.to(device)
                
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _generate_response(self, inputs: Dict[str, Any], text: str, module_name: str) -> str:
        """Generate response using the local model"""
        try:
            import time
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] LOCAL HF VLM QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Start timing
            generation_start = time.time()
            
            with self.torch.no_grad():
                # Ensure all inputs are on the correct device
                if hasattr(self.model, 'device'):
                    device = self.model.device
                elif hasattr(self.model, 'module') and hasattr(self.model.module, 'device'):
                    device = self.model.module.device
                else:
                    device = next(self.model.parameters()).device
                
                # Move inputs to device if needed
                inputs_on_device = {k: v.to(device) for k, v in inputs.items()}

                # Set termination condition for generation
                eos_token_id = self.processor.tokenizer.eos_token_id
                
                generated_ids = self.model.generate(
                    **inputs_on_device,
                    max_new_tokens=256,  # Reduced for faster JSON generation
                    do_sample=True, # Enable sampling for variety
                    temperature=0.7, # Add some randomness
                    top_p=0.9, # Nucleus sampling
                    eos_token_id=eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                # Decode the response, removing the prompt part
                input_token_len = inputs_on_device["input_ids"].shape[1]
                generated_text = self.processor.batch_decode(
                    generated_ids[:, input_token_len:],
                    skip_special_tokens=True
                )[0]
                
                # Clean up the output
                result = generated_text.strip()

            # End timing and log performance
            generation_end = time.time()
            generation_time = generation_end - generation_start
            
            # Log the response with timing
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ⏱️  GENERATION TIME: {generation_time:.3f} seconds")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def get_query(self, img: Union[Image.Image, np.ndarray, str], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using local HuggingFace model"""
        from PIL import Image
        
        # Handle PIL Images, numpy arrays, AND string file paths
        if isinstance(img, str):
            # It's a file path - load it
            image = Image.open(img).convert("RGB")
        elif hasattr(img, 'convert'):  # It's a PIL Image
            image = img.convert("RGB")
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Handle different model formats using stored model type
        if self.model_type == "qwen2_vl":
            # Qwen2-VL format - must use chat template for image processing
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
        else:
            # Phi-3 Vision format - keep existing format for compatibility
            prompt = f"<|user|>\n<|image_1|>\n{text}<|end|>\n<|assistant|>\n"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        return self._generate_response(inputs, prompt, module_name)
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using local HuggingFace model"""
        try:
            # Use chat template format for consistency with image processing
            messages = [{"role": "user", "content": text}]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process the prompt 
            inputs = self.processor(text=prompt, return_tensors="pt")
            
            return self._generate_response(inputs, prompt, module_name)
        except Exception as e:
            # Fallback to direct text if chat template fails
            logger.warning(f"Chat template failed for text query, falling back to direct text: {e}")
            inputs = self.processor(text=text, return_tensors="pt")
            return self._generate_response(inputs, text, module_name)

class LegacyOllamaBackend(VLMBackend):
    """Legacy Ollama backend for backward compatibility"""
    
    def __init__(self, model_name: str, port: int = 8010, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install with: pip install openai")
        
        self.model_name = model_name
        self.port = port
        self.client = OpenAI(api_key='', base_url=f'http://localhost:{port}/v1')
    
    @retry_with_exponential_backoff
    def _call_completion(self, messages):
        """Calls the completions.create method with exponential backoff."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using legacy Ollama backend"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            image = img
        elif hasattr(img, 'shape'):  # It's a numpy array
            image = Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OLLAMA VLM IMAGE QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using legacy Ollama backend"""
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": text}]
        }]
        
        # Log the prompt
        prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
        logger.info(f"[{module_name}] OLLAMA VLM TEXT QUERY:")
        logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
        
        response = self._call_completion(messages)
        result = response.choices[0].message.content
        
        # Log the response
        result_preview = result[:1000] + "..." if len(result) > 1000 else result
        logger.info(f"[{module_name}] RESPONSE: {result_preview}")
        logger.info(f"[{module_name}] ---")
        
        return result

class VertexBackend(VLMBackend):
    """Google Gemini API with Vertex backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            from google import genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")
        
        self.model_name = model_name
        
        # Initialize the model
        self.client = genai.Client(
            vertexai=True,
            project='pokeagent-011',
            location='us-central1',
        )
        self.genai = genai
        
        logger.info(f"Gemini backend initialized with model: {model_name}")
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=content_parts
        )
        return response
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using Gemini API"""
        try:
            image = self._prepare_image(img)
            
            # Prepare content for Gemini
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)
            
            result = response.text
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini image query: {e}")
            # Don't attempt text fallback on TimeoutError — the SIGALRM has already fired
            # and the fallback call would hang indefinitely with no new timeout set.
            if isinstance(e, TimeoutError):
                raise
            # Try text-only fallback for other Gemini errors
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using Gemini API"""
        try:
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content([text])
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."
            
            result = response.text
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."


class GeminiBackend(VLMBackend):
    """Google Gemini API backend"""
    
    def __init__(self, model_name: str, **kwargs):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Google Generative AI package not found. Install with: pip install google-generativeai")
        
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Error: Gemini API key is missing! Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        
        # Configure the API with explicit API key (not ADC)
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(model_name)
        self.genai = genai
        
        logger.info(f"Gemini backend initialized with model: {model_name}")
    
    def _prepare_image(self, img: Union[Image.Image, np.ndarray]) -> Image.Image:
        """Prepare image for Gemini API"""
        # Handle both PIL Images and numpy arrays
        if hasattr(img, 'convert'):  # It's a PIL Image
            return img
        elif hasattr(img, 'shape'):  # It's a numpy array
            return Image.fromarray(img)
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
    
    @retry_with_exponential_backoff
    def _call_generate_content(self, content_parts):
        """Calls the generate_content method with exponential backoff."""
        response = self.model.generate_content(content_parts)
        response.resolve()
        return response
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt using Gemini API"""
        start_time = time.time()
        try:
            image = self._prepare_image(img)
            
            # Prepare content for Gemini
            content_parts = [text, image]
            
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM IMAGE QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content(content_parts)
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Trying text-only fallback.")
                    # Fallback to text-only query
                    return self.get_text_query(text, module_name)
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": True, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini image query: {e}")
            # Don't attempt text fallback on TimeoutError — the SIGALRM has already fired
            # and the fallback call would hang indefinitely with no new timeout set.
            if isinstance(e, TimeoutError):
                raise
            # Try text-only fallback for other Gemini errors
            try:
                logger.info(f"[{module_name}] Attempting text-only fallback due to error: {e}")
                return self.get_text_query(text, module_name)
            except Exception as fallback_error:
                logger.error(f"[{module_name}] Text-only fallback also failed: {fallback_error}")
                raise e
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt using Gemini API"""
        start_time = time.time()
        try:
            # Log the prompt
            prompt_preview = text[:2000] + "..." if len(text) > 2000 else text
            logger.info(f"[{module_name}] GEMINI VLM TEXT QUERY:")
            logger.info(f"[{module_name}] PROMPT: {prompt_preview}")
            
            # Generate response
            response = self._call_generate_content([text])
            
            # Check for safety filter or content policy issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                    logger.warning(f"[{module_name}] Gemini safety filter triggered (finish_reason=12). Returning default response.")
                    return "I cannot analyze this content due to safety restrictions. I'll proceed with a basic action: press 'A' to continue."
            
            result = response.text
            duration = time.time() - start_time
            
            # Extract token usage if available
            token_usage = {}
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                token_usage = {
                    "prompt_tokens": getattr(usage, 'prompt_token_count', 0),
                    "completion_tokens": getattr(usage, 'candidates_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0)
                }
            
            # Log the interaction
            log_llm_interaction(
                interaction_type=f"gemini_{module_name}",
                prompt=text,
                response=result,
                duration=duration,
                metadata={"model": self.model_name, "backend": "gemini", "has_image": False, "token_usage": token_usage},
                model_info={"model": self.model_name, "backend": "gemini"}
            )
            
            # Log the response
            result_preview = result[:1000] + "..." if len(result) > 1000 else result
            logger.info(f"[{module_name}] RESPONSE: {result_preview}")
            logger.info(f"[{module_name}] ---")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini text query: {e}")
            # Return a safe default response
            logger.warning(f"[{module_name}] Returning default response due to error: {e}")
            return "I encountered an error processing the request. I'll proceed with a basic action: press 'A' to continue."

class VLM:
    """Main VLM class that supports multiple backends"""
    
    BACKENDS = {
        'openai': OpenAIBackend,
        'openrouter': OpenRouterBackend,
        'local': LocalHuggingFaceBackend,
        'gemini': GeminiBackend,
        'ollama': LegacyOllamaBackend,  # Legacy support
        'vertex': VertexBackend,  # Added Vertex backend
    }
    
    def __init__(self, model_name: str, backend: str = 'openai', port: int = 8010, **kwargs):
        """
        Initialize VLM with specified backend
        
        Args:
            model_name: Name of the model to use
            backend: Backend type ('openai', 'openrouter', 'local', 'gemini', 'ollama')
            port: Port for Ollama backend (legacy)
            **kwargs: Additional arguments passed to backend
        """
        self.model_name = model_name
        self.backend_type = backend.lower()
        
        # Auto-detect backend based on model name if not explicitly specified
        if backend == 'auto':
            self.backend_type = self._auto_detect_backend(model_name)
        
        if self.backend_type not in self.BACKENDS:
            raise ValueError(f"Unsupported backend: {self.backend_type}. Available: {list(self.BACKENDS.keys())}")
        
        # Initialize the appropriate backend
        backend_class = self.BACKENDS[self.backend_type]
        
        # Pass port parameter for legacy Ollama backend
        if self.backend_type == 'ollama':
            self.backend = backend_class(model_name, port=port, **kwargs)
        else:
            self.backend = backend_class(model_name, **kwargs)
        
        logger.info(f"VLM initialized with {self.backend_type} backend using model: {model_name}")
    
    def _auto_detect_backend(self, model_name: str) -> str:
        """Auto-detect backend based on model name"""
        model_lower = model_name.lower()
        
        if any(x in model_lower for x in ['gpt', 'o4-mini', 'o3', 'claude']):
            return 'openai'
        elif any(x in model_lower for x in ['gemini', 'palm']):
            return 'gemini'
        elif any(x in model_lower for x in ['llama', 'mistral', 'qwen', 'phi']):
            return 'local'
        else:
            # Default to OpenAI for unknown models
            return 'openai'
    
    def get_query(self, img: Union[Image.Image, np.ndarray], text: str, module_name: str = "Unknown") -> str:
        """Process an image and text prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_query(img, text, module_name)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": True}
            )
            raise
    
    def get_text_query(self, text: str, module_name: str = "Unknown") -> str:
        """Process a text-only prompt"""
        try:
            # Backend handles its own logging, so we don't duplicate it here
            result = self.backend.get_text_query(text, module_name)
            return result
        except Exception as e:
            # Only log errors that aren't already logged by the backend
            duration = 0  # Backend tracks actual duration
            log_llm_error(
                interaction_type=f"{self.backend.__class__.__name__.lower()}_{module_name}",
                prompt=text,
                error=str(e),
                metadata={"model": self.model_name, "backend": self.backend.__class__.__name__, "duration": duration, "has_image": False}
            )
            raise
