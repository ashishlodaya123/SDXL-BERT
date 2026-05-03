# -*- coding: utf-8 -*-
"""
Enhanced SDXL Pipeline with T5+BERT, Summarization, Refined Prompting,
High CLIP Scoring, LCM Refiner & CLIP Interrogator Hybrid Score (v8)

This script implements an advanced text-to-image generation pipeline aiming for
consistently high CLIP scores through improved prompt adherence,
refined scoring logic, flexible generation parameters, LCM-based refinement,
and hybrid CLIP scoring using reverse captioning.

Key Improvements in v8 (LCM & Hybrid Score Focus):
1.  **LCM Refiner:** Replaces the standard SDXL Img2Img refiner with the
    faster `latent-consistency/lcm-sdxl` model for refinement, allowing
    fewer refiner steps (e.g., 4-8).
2.  **CLIP Interrogator Integration:** After image generation, uses the
    `clip-interrogator` library (BLIP+CLIP) to generate a reverse caption
    for each candidate image.
3.  **Hybrid CLIP Score:** Calculates a final score based on the average of:
    a) The original multi-prompt CLIP score (image vs. original/optimized prompts).
    b) A "reverse" CLIP score (image vs. its generated reverse caption).
    This hybrid score is used for selecting the best candidate.
4.  **Configurable LCM Steps:** Added `--lcm_refiner_steps` argument.
5.  **Optional CLIP Interrogation:** Added `--enable_clip_interrogation` flag.
6.  Maintains all features from v7 (normalized CLIP, original prompt scoring, etc.).
"""

import torch
# --- v8 Change: Import LCM Pipeline ---
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
try:
    # Use LCM pipeline if available
    from diffusers import DiffusionPipeline # General pipeline loader
    # Specific LCM pipeline class (adjust path if library structure changes)
    # Assuming it might be directly loadable via DiffusionPipeline or has a specific class
    # Let's try loading via DiffusionPipeline first, fallback later if needed
    # Note: The original request specified `lcm.lcm_sdxl_img2img.LCMStableDiffusionXLImg2ImgPipeline`
    # which might be from a specific fork or older version. Using the standard diffusers
    # approach is generally preferred if the model is hosted on HF Hub.
    LCM_REFINER_MODEL = "latent-consistency/lcm-sdxl"
    LCM_AVAILABLE = True
except ImportError:
    logging.warning("LCMStableDiffusionXLImg2ImgPipeline not found. Falling back to standard SDXL Refiner.")
    LCM_AVAILABLE = False
    LCM_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0" # Fallback

from transformers import (T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration,
                         CLIPTokenizer, CLIPProcessor, CLIPModel, CLIPConfig,
                         BertTokenizer, BertModel, pipeline as hf_pipeline)
import spacy
import re
from collections import defaultdict
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import time
from functools import lru_cache
import gc
import os
import json
import random
from tqdm import tqdm
import sys
import subprocess
import argparse
import logging
import math
from typing import Dict, List, Tuple, Optional, Union

# --- v8 Change: Import CLIP Interrogator ---
try:
    from clip_interrogator import Config as ClipInterrogatorConfig
    from clip_interrogator import Interrogator as ClipInterrogator
    CLIP_INTERROGATOR_AVAILABLE = True
except ImportError:
    logging.warning("`clip-interrogator` library not found. Hybrid scoring will be disabled.")
    CLIP_INTERROGATOR_AVAILABLE = False
    ClipInterrogatorConfig = None
    ClipInterrogator = None


# --- Configuration & Constants ---

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Model Configuration
SDXL_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
# --- v8 Change: Use LCM model path if available ---
SDXL_REFINER_MODEL = LCM_REFINER_MODEL if LCM_AVAILABLE else "stabilityai/stable-diffusion-xl-refiner-1.0"

# NLP & CLIP Models
SPACY_MODEL_NAME = "en_core_web_lg"
T5_ENCODER_MODEL_NAME = 't5-small'
T5_SUMMARIZER_MODEL_NAME = 'google/flan-t5-base'
BERT_MODEL_NAME = 'bert-base-uncased'
CLIP_SCORING_MODEL = "openai/clip-vit-large-patch14"
CLIP_TOKENIZER_PATH = "openai/clip-vit-large-patch14"
# --- v8 Change: CLIP Interrogator Config ---
CLIP_INTERROGATOR_MODEL_NAME = "ViT-L-14/openai" # Model used by interrogator

# Generation Parameters
DEFAULT_TOKEN_LIMIT = 77
MAX_PROMPT_WEIGHT = 1.6
MIN_PROMPT_WEIGHT = 0.85
BASE_STEPS = 40
COMPLEXITY_STEP_BOOST = 15
DEFAULT_GUIDANCE_SCALE = 7.5
GUIDANCE_RANDOM_RANGE = (6.5, 9.0)
DEFAULT_REFINER_STRENGTH = 0.25 # Strength for LCM/Standard Refiner
# --- v8 Change: Default steps for LCM vs Standard Refiner ---
DEFAULT_LCM_REFINER_STEPS = 6 # Fewer steps needed for LCM
DEFAULT_STD_REFINER_STEPS_RATIO = 0.25 # Ratio for standard refiner
DEFAULT_ENHANCE_LEVEL = 0.6
DEFAULT_NEGATIVE_PROMPT = "low quality, bad anatomy, distorted, blurry, watermark, cropped, out of frame, NSFW, text, words, letters, signature, username, jpeg artifacts, noisy, unclear, mutated, deformed, disfigured, duplicate, morbid, mutilated, extra limbs, extra fingers, poorly drawn hands, poorly drawn feet, poorly drawn face, long neck, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, tiling, worst quality, lowres, low detail, error"


# Prompt Optimization Parameters
SUMMARIZATION_TRIGGER_LENGTH = 90
SUMMARIZATION_MAX_LENGTH = 100
SUMMARIZATION_MIN_LENGTH = 40

# CLIP Scoring & Candidate Generation
DEFAULT_NUM_CANDIDATES = 8
MIN_CLIP_SCORE_THRESHOLD = 0.6 # For original CLIP score part

# --- Utility Functions ---

def get_device():
    """Determines the optimal device for Torch."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logging.warning("Using MPS device (macOS). Ensure compatibility and performance.")
        return torch.device("mps")
    else:
        logging.info("CUDA/MPS not available, using CPU.")
        return torch.device("cpu")

DEVICE = get_device()
logging.info(f"Selected compute device: {DEVICE}")

def load_spacy_model(model_name=SPACY_MODEL_NAME):
    """Loads a spaCy model, downloading if necessary."""
    try:
        logging.info(f"Loading spaCy model: {model_name}")
        return spacy.load(model_name, disable=["ner", "textcat"])
    except OSError:
        logging.warning(f"SpaCy model '{model_name}' not found. Downloading...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            return spacy.load(model_name, disable=["ner", "textcat"])
        except Exception as e:
            logging.error(f"Failed to download or load spaCy model: {e}")
            raise RuntimeError(f"Could not load spaCy model '{model_name}'.")

try:
    nlp = load_spacy_model()
except RuntimeError as e:
    logging.error(f"Fatal error loading spaCy: {e}. Exiting.")
    sys.exit(1)

def release_memory(*args):
    """Releases CUDA/MPS cache and runs GC, deleting provided objects."""
    for obj in args:
        try:
            del obj
        except NameError:
            pass
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

# --- v8 Change: CLIP Interrogator Helper ---
_clip_interrogator_instance = None # Global instance to avoid reloading

def get_clip_interrogator(device=DEVICE):
    """Initializes and returns a singleton CLIP Interrogator instance."""
    global _clip_interrogator_instance
    if not CLIP_INTERROGATOR_AVAILABLE:
        logging.warning("CLIP Interrogator library not available.")
        return None

    if _clip_interrogator_instance is None:
        logging.info(f"Initializing CLIP Interrogator (Model: {CLIP_INTERROGATOR_MODEL_NAME})...")
        try:
            # Determine device for interrogator (prefer CUDA/MPS if available)
            ci_device = "cuda" if device.type == "cuda" else "mps" if device.type == "mps" else "cpu"
            ci_config = ClipInterrogatorConfig(
                clip_model_name=CLIP_INTERROGATOR_MODEL_NAME,
                device=ci_device,
                # quiet=True # Suppress internal logging if desired
            )
            _clip_interrogator_instance = ClipInterrogator(ci_config)
            logging.info("CLIP Interrogator initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize CLIP Interrogator: {e}", exc_info=True)
            _clip_interrogator_instance = None # Ensure it's None on failure
    return _clip_interrogator_instance

def run_clip_interrogation(image: Image.Image, mode: str = 'fast') -> Optional[str]:
    """
    Generates a caption for an image using CLIP Interrogator.

    Args:
        image: The PIL Image to caption.
        mode: 'fast' or 'best' mode for interrogation.

    Returns:
        The generated caption string, or None if interrogation fails.
    """
    interrogator = get_clip_interrogator()
    if interrogator is None or image is None:
        return None

    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        logging.debug(f"Running CLIP Interrogation (mode: {mode})...")
        if mode == 'fast':
            caption = interrogator.interrogate_fast(image)
        elif mode == 'best':
            caption = interrogator.interrogate(image) # Slower, potentially better
        else:
             logging.warning(f"Unknown interrogation mode '{mode}', using 'fast'.")
             caption = interrogator.interrogate_fast(image)

        logging.debug(f"CLIP Interrogation result: {caption}")
        return caption.strip() if caption else None

    except Exception as e:
        logging.error(f"CLIP Interrogation failed: {e}", exc_info=False) # Keep log concise
        # Handle potential OOM
        if "memory" in str(e).lower():
             logging.warning("Potential OOM during CLIP Interrogation. Consider releasing memory.")
             release_clip_interrogator_memory()
             release_memory()
        return None

def release_clip_interrogator_memory():
    """Releases the CLIP Interrogator instance and memory."""
    global _clip_interrogator_instance
    if _clip_interrogator_instance is not None:
        logging.info("Releasing CLIP Interrogator resources...")
        interrogator_ref = _clip_interrogator_instance
        _clip_interrogator_instance = None
        # Explicitly delete and clear cache (may depend on library internals)
        release_memory(interrogator_ref)
        logging.info("CLIP Interrogator resources released.")

# --- v8 Change: LCM Refiner Loading Helper ---
def load_lcm_refiner(base_pipe: StableDiffusionXLPipeline, device: torch.device, dtype: torch.dtype, variant: Optional[str]) -> Optional[DiffusionPipeline]:
    """Loads the LCM-SDXL refiner, sharing components with the base pipe."""
    if not LCM_AVAILABLE:
        logging.warning("LCM library components not found, cannot load LCM refiner.")
        return None

    logging.info(f"Loading LCM-SDXL Refiner ({LCM_REFINER_MODEL})...")
    try:
        # Load using DiffusionPipeline.from_pretrained, which should handle LCM models
        # Share VAE and Text Encoder 2 for efficiency and consistency
        lcm_refiner = DiffusionPipeline.from_pretrained(
            LCM_REFINER_MODEL,
            vae=base_pipe.vae,
            text_encoder_2=base_pipe.text_encoder_2,
            torch_dtype=dtype,
            variant=variant if dtype == torch.float16 else None,
            use_safetensors=True # Prefer safetensors
        ).to(device)
        logging.info("LCM-SDXL Refiner loaded successfully.")
        return lcm_refiner
    except Exception as e:
        logging.error(f"Failed to load LCM-SDXL Refiner: {e}", exc_info=True)
        # Fallback or raise error? For now, return None, pipeline will handle it.
        return None


# --- Core Classes (Mostly unchanged, except for CLIPScorer and Pipeline) ---

class T5SemanticAnalyzer:
    """Uses a T5 encoder model for semantic embedding generation."""
    def __init__(self, model_name=T5_ENCODER_MODEL_NAME, device=DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.initialized = False
        self.embedding_cache = {}

    def initialize(self):
        if not self.initialized:
            logging.info(f"Initializing T5 semantic analyzer ({self.model_name})...")
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5EncoderModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                self.initialized = True
                logging.info("T5 semantic analyzer initialized successfully.")
            except Exception as e:
                logging.error(f"Error initializing T5 encoder model: {e}")
                self.initialized = False

    @lru_cache(maxsize=2048)
    def get_embedding(self, text):
        self.initialize()
        if not self.initialized:
            return torch.zeros(512, device='cpu')
        clean_text = str(text).lower().strip()
        if not clean_text: return torch.zeros(512, device='cpu')
        if clean_text in self.embedding_cache: return self.embedding_cache[clean_text]
        try:
            inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad(): outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            embedding_cpu = embedding.cpu()
            self.embedding_cache[clean_text] = embedding_cpu
            return embedding_cpu
        except Exception as e:
            logging.error(f"Error getting T5 embedding for '{clean_text[:50]}...': {e}")
            return torch.zeros(512, device='cpu')

    def release_memory(self):
        if self.initialized:
            model_ref, tokenizer_ref = self.model, self.tokenizer
            self.model, self.tokenizer, self.initialized = None, None, False
            self.embedding_cache.clear(); release_memory(model_ref, tokenizer_ref)
            logging.info("T5 semantic analyzer resources released.")

class DualSemanticAnalyzer:
    """Uses both T5 and BERT models for semantic analysis."""
    def __init__(self, t5_encoder_model=T5_ENCODER_MODEL_NAME, bert_model=BERT_MODEL_NAME, device=DEVICE):
        self.device = device
        self.t5_analyzer = T5SemanticAnalyzer(t5_encoder_model, device)
        self.bert_model_name = bert_model
        self.bert_tokenizer, self.bert_model = None, None
        self.initialized = False
        self.embedding_cache, self.similarity_cache = {}, {}
        self.t5_emb_size, self.bert_emb_size = 512, 768
        self.combined_emb_size = self.bert_emb_size

    def initialize(self):
        if not self.initialized:
            logging.info("Initializing dual semantic analyzer (T5 + BERT)...")
            self.t5_analyzer.initialize()
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = BertModel.from_pretrained(self.bert_model_name).to(self.device)
                self.bert_model.eval()
                if self.t5_analyzer.initialized: self.initialized = True; logging.info("Dual semantic analyzer initialized successfully.")
                else: self.initialized = True; logging.warning("Dual analyzer initialized (BERT only) as T5 failed.")
            except Exception as e: logging.error(f"Error initializing BERT model: {e}"); self.initialized = False

    @lru_cache(maxsize=2048)
    def get_embedding(self, text):
        self.initialize()
        if not self.initialized: return torch.zeros(self.combined_emb_size, device='cpu')
        clean_text = str(text).lower().strip();
        if not clean_text: return torch.zeros(self.combined_emb_size, device='cpu')
        if clean_text in self.embedding_cache: return self.embedding_cache[clean_text]
        try:
            t5_emb, bert_emb = None, None
            if self.t5_analyzer.initialized: t5_emb = self.t5_analyzer.get_embedding(clean_text); t5_emb = t5_emb.to(self.device) if t5_emb is not None else None
            if self.bert_model is not None:
                bert_inputs = self.bert_tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                bert_inputs = {k: v.to(self.device) for k, v in bert_inputs.items()}
                with torch.no_grad(): bert_outputs = self.bert_model(**bert_inputs); bert_emb = bert_outputs.last_hidden_state.mean(dim=1).squeeze()

            combined_emb = torch.zeros(self.combined_emb_size, device=self.device)
            t5_available = t5_emb is not None and torch.is_tensor(t5_emb) and t5_emb.numel() > 0
            bert_available = bert_emb is not None and torch.is_tensor(bert_emb) and bert_emb.numel() > 0

            if t5_available and bert_available:
                pad_size = self.bert_emb_size - self.t5_emb_size; t5_pad = torch.nn.functional.pad(t5_emb, (0, pad_size)) if pad_size >= 0 else t5_emb[:self.bert_emb_size]
                t5_norm, bert_norm = torch.norm(t5_pad), torch.norm(bert_emb)
                if t5_norm > 1e-6 and bert_norm > 1e-6: combined_emb = (t5_pad / t5_norm + bert_emb / bert_norm) / 2.0
                elif bert_norm > 1e-6: combined_emb = bert_emb / bert_norm
                elif t5_norm > 1e-6: combined_emb = t5_pad / t5_norm
            elif bert_available: bert_norm = torch.norm(bert_emb); combined_emb = bert_emb / bert_norm if bert_norm > 1e-6 else torch.zeros_like(bert_emb)
            elif t5_available:
                pad_size = self.bert_emb_size - self.t5_emb_size; t5_pad = torch.nn.functional.pad(t5_emb, (0, pad_size)) if pad_size >= 0 else t5_emb[:self.bert_emb_size]
                t5_norm = torch.norm(t5_pad); combined_emb = t5_pad / t5_norm if t5_norm > 1e-6 else torch.zeros(self.combined_emb_size, device=self.device)

            embedding_cpu = combined_emb.cpu(); self.embedding_cache[clean_text] = embedding_cpu; return embedding_cpu
        except Exception as e: logging.error(f"Error getting dual embedding for '{clean_text[:50]}...': {e}"); return torch.zeros(self.combined_emb_size, device='cpu')

    def get_similarity(self, text1, text2):
        key = tuple(sorted((str(text1).lower().strip(), str(text2).lower().strip())))
        if key in self.similarity_cache: return self.similarity_cache[key]
        if not key[0] or not key[1]: return 0.0
        try:
            emb1, emb2 = self.get_embedding(key[0]), self.get_embedding(key[1])
            if not torch.is_tensor(emb1) or not torch.is_tensor(emb2) or emb1.shape != emb2.shape: return 0.0
            emb1, emb2 = emb1.to(self.device), emb2.to(self.device)
            norm1, norm2 = torch.norm(emb1), torch.norm(emb2)
            if norm1 < 1e-6 or norm2 < 1e-6: return 0.0
            similarity = torch.dot(emb1 / norm1, emb2 / norm2).item()
            similarity = max(0.0, min(similarity, 1.0)); self.similarity_cache[key] = similarity; return similarity
        except Exception as e: logging.warning(f"Error calculating similarity between '{key[0][:50]}' and '{key[1][:50]}': {e}"); return 0.0

    def analyze_element_importance(self, element_name, context_sentences, full_text):
        self.initialize()
        if not self.initialized: return 0.5
        element_lower = element_name.lower(); element_parts = set(element_lower.split())
        relevant_sentences = [s for s in context_sentences if any(part in s.lower() for part in element_parts)] or context_sentences
        direct_relevance = self.get_similarity(element_name, full_text)
        if not relevant_sentences: return direct_relevance * 0.9
        sentence_relevances = [self.get_similarity(sent, full_text) for sent in relevant_sentences]
        avg_sentence_relevance = sum(sentence_relevances) / len(sentence_relevances) if sentence_relevances else 0.0
        element_to_context_relevance = [self.get_similarity(element_name, sent) for sent in relevant_sentences]
        avg_element_context_relevance = sum(element_to_context_relevance) / len(element_to_context_relevance) if element_to_context_relevance else 0.0
        combined = (avg_sentence_relevance * 0.2) + (direct_relevance * 0.4) + (avg_element_context_relevance * 0.4)
        return max(0.0, min(combined, 1.0))

    def release_memory(self):
        self.t5_analyzer.release_memory()
        if self.bert_model is not None:
            bert_ref, tokenizer_ref = self.bert_model, self.bert_tokenizer
            self.bert_model, self.bert_tokenizer = None, None
            release_memory(bert_ref, tokenizer_ref); logging.info("BERT resources released.")
        self.initialized = False; self.embedding_cache.clear(); self.similarity_cache.clear()
        logging.info("Dual semantic analyzer resources released.")


class CLIPScorer:
    """Enhanced CLIP scorer with normalization and multi-prompt scoring."""
    def __init__(self, model_name=CLIP_SCORING_MODEL, device=DEVICE):
        self.model_name = model_name
        self.device = self._select_device(device)
        self.processor, self.model, self.config = None, None, None
        self.max_length = 77
        self.initialized = False

    def _select_device(self, requested_device):
        if requested_device.type == "cuda" and torch.cuda.is_available():
             try: torch.cuda.get_device_name(0); logging.info("CLIP using CUDA device."); return requested_device
             except Exception as e: logging.warning(f"CUDA device check failed ({e}), falling back to CPU for CLIP."); return torch.device("cpu")
        elif requested_device.type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
             logging.info("CLIP using MPS device."); return requested_device
        else: logging.info("CLIP using CPU device."); return torch.device("cpu")

    def initialize(self):
        if not self.initialized:
            logging.info(f"Initializing CLIP scorer ({self.model_name}) on device {self.device}...")
            try:
                self.config = CLIPConfig.from_pretrained(self.model_name)
                self.processor = CLIPProcessor.from_pretrained(self.model_name)
                self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                try: self.max_length = self.config.text_config.max_position_embeddings
                except AttributeError: logging.warning("Could not determine max_length from CLIP config, using default 77."); self.max_length = 77
                self.initialized = True; logging.info("CLIP scorer initialized successfully.")
            except Exception as e: logging.error(f"Error initializing CLIP model: {e}"); self._reset_state()

    def _reset_state(self):
        self.processor, self.model, self.config, self.initialized = None, None, None, False; self.max_length = 77

    def calculate_clip_score(self, image: Image.Image, text: str) -> float:
        """Calculates CLIP score [0, 1]. Returns 0.0 on error."""
        self.initialize()
        if not self.initialized or image is None or not text: return 0.0
        try:
            if image.mode != 'RGB': image = image.convert('RGB')
            inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_features, text_features = outputs.image_embeds, outputs.text_embeds
                image_features_norm = torch.nn.functional.normalize(image_features, p=2, dim=-1)
                text_features_norm = torch.nn.functional.normalize(text_features, p=2, dim=-1)
                similarity = torch.matmul(image_features_norm, text_features_norm.T).item()
            normalized_score = (similarity + 1.0) / 2.0
            final_score = max(0.0, min(normalized_score, 1.0))
            return final_score
        except Exception as e:
            logging.error(f"CLIP scoring error for text '{text[:50]}...': {e}")
            if "memory" in str(e).lower(): logging.warning("Potential OOM during CLIP scoring."); self.release_memory(); release_memory()
            return 0.0

    def calculate_multi_prompt_score(self, image: Image.Image, prompts: List[str]) -> Tuple[float, Dict[str, float]]:
        """Calculates average CLIP score across multiple prompts [0, 1]."""
        if not prompts or image is None: return 0.0, {}
        scores_dict = {}
        valid_prompts = [p for p in prompts if p and isinstance(p, str)]
        if not valid_prompts: return 0.0, {}
        for prompt in valid_prompts:
            score = self.calculate_clip_score(image, prompt)
            scores_dict[prompt] = score
        if not scores_dict: return 0.0, {}
        average_score = sum(scores_dict.values()) / len(scores_dict)
        return average_score, scores_dict

    def release_memory(self):
        if self.initialized:
            model_ref, processor_ref, config_ref = self.model, self.processor, self.config
            self._reset_state(); release_memory(model_ref, processor_ref, config_ref)
            logging.info("CLIP scorer resources released.")


class DynamicPromptOptimizer:
    """
    Enhanced prompt optimizer (v7 logic - summarization, dual semantics, weighting, etc.).
    Unchanged from v7 as core logic remains the same for v8.
    """
    def __init__(self, token_limit=DEFAULT_TOKEN_LIMIT,
                 summarizer_model=T5_SUMMARIZER_MODEL_NAME, device=DEVICE):
        self.device = device
        self.token_limit = token_limit
        self.summarization_trigger_length = SUMMARIZATION_TRIGGER_LENGTH
        self.summarization_max_length = SUMMARIZATION_MAX_LENGTH
        self.summarization_min_length = SUMMARIZATION_MIN_LENGTH
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_PATH)
            logging.info(f"CLIP Tokenizer loaded from {CLIP_TOKENIZER_PATH} for prompt length calculation.")
        except Exception as e:
            logging.error(f"Error loading CLIP tokenizer: {e}. Prompt token counting may be inaccurate.")
            self.tokenizer = None
        self.semantic_analyzer = DualSemanticAnalyzer(device=device)
        self.summarizer_model_name = summarizer_model
        self.summarizer, self.summarizer_initialized, self.summarizer_tokenizer = None, False, None
        self.quality_style_terms = {
            "ultra realistic": 1.6, "hyperrealistic": 1.6, "photorealistic": 1.5, "8k uhd": 1.5,
            "high resolution": 1.4, "ultra detailed": 1.5, "photorealism": 1.5, "intricate details": 1.4,
            "sharp focus": 1.4, "extremely detailed": 1.4, "cinematic": 1.4, "dramatic": 1.3, "epic": 1.3,
            "dslr photo": 1.4, "professional photography": 1.4, "shot on hasselblad": 1.35, "film grain": 1.2,
            "depth of field": 1.3, "bokeh": 1.2, "long exposure": 1.2, "cinematic lighting": 1.5,
            "dramatic lighting": 1.4, "god rays": 1.35, "volumetric lighting": 1.3, "golden hour": 1.3,
            "moody lighting": 1.3, "masterpiece": 1.3, "best quality": 1.2, "high quality": 1.1,
            "trending on artstation": 1.2, "oil painting": 1.15, "watercolor": 1.1, "digital painting": 1.1,
            "concept art": 1.2, "illustration": 1.1, "sketch": 1.05, "anime style": 1.1,
            "unreal engine": 1.3, "octane render": 1.3
        }
        self.color_terms = {"red", "blue", "green", "yellow", "orange", "purple", "pink", "violet", "brown", "black", "white", "gray", "grey", "silver", "gold", "cyan", "magenta", "turquoise", "maroon", "navy", "olive", "teal", "beige", "scarlet", "indigo", "lime", "amber", "crimson", "azure"}
        self.background_keywords = {"background", "setting", "environment", "scene", "landscape", "cityscape", "room", "space", "sky", "ground", "horizon", "floor", "backdrop", "field", "forest", "mountain", "ocean", "sea", "beach", "underwater", "outer space", "galaxy", "nebula", "indoors", "outdoors"}
        self.ignore_elements = {
            "a", "an", "the", "of", "in", "on", "at", "by", "for", "to", "from", "with", "and", "or", "but", "so", "as", "it", "is", "was", "were", "be", "being", "been", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "i", "you", "he", "she", "we", "they", "have", "has", "had", "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must", "very", "highly", "extremely", "really", "truly", "quite", "rather", "somewhat", "good", "bad", "nice", "beautiful", "amazing", "stunning", "great", "wonderful", "large", "small", "big", "little", "huge", "tiny", "some", "many", "several", "various", "different", "other", "another", "few", "all", "photo", "image", "picture", "photograph", "art", "artwork", "style", "graphic", "quality", "resolution", "render", "rendering", "illustration", "drawing", "painting", "realistic", "photorealistic", "hyperrealistic", "detailed", "intricate", "complex", "cinematic", "dramatic", "epic", "background", "foreground", "setting", "scene", "view", "looking", "show", "depict", "feature", "featuring", "contain", "including", "made", "create", "generate", "wearing", "holding", "sitting", "standing", "lying", "focus", "closeup", "wide shot", "full body", "portrait", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
        } | spacy.lang.en.stop_words.STOP_WORDS

    def _initialize_summarizer(self):
        if not self.summarizer_initialized:
            logging.info(f"Initializing T5 summarizer ({self.summarizer_model_name})...")
            try:
                self.summarizer_tokenizer = T5Tokenizer.from_pretrained(self.summarizer_model_name)
                summarizer_model = T5ForConditionalGeneration.from_pretrained(self.summarizer_model_name)
                device_index = self.device.index if self.device.type == "cuda" else -1
                self.summarizer = hf_pipeline("summarization", model=summarizer_model, tokenizer=self.summarizer_tokenizer, device=device_index)
                self.summarizer_initialized = True; logging.info(f"T5 summarizer initialized on device {self.device if device_index == -1 else f'cuda:{device_index}'}.")
            except Exception as e:
                logging.error(f"Error initializing T5 summarizer pipeline: {e}")
                self.summarizer, self.summarizer_tokenizer, self.summarizer_initialized = None, None, False

    def _summarize_prompt(self, text):
        self._initialize_summarizer()
        if not self.summarizer or not text: return text
        try:
            logging.info(f"Summarizing prompt (original length: {len(text.split())} words)...")
            input_text = "summarize: " + text; input_word_count = len(text.split())
            max_len = min(self.summarization_max_length, max(self.summarization_min_length + 15, int(input_word_count * 0.7)))
            min_len = min(max_len - 10, max(self.summarization_min_length, int(input_word_count * 0.25)))
            if min_len >= max_len: min_len = max(10, max_len - 15)
            logging.debug(f"Summarization constraints: min_length={min_len}, max_length={max_len}")
            summary_result = self.summarizer(input_text, max_length=max_len, min_length=min_len, num_beams=4, early_stopping=True, do_sample=False)
            if summary_result and isinstance(summary_result, list) and 'summary_text' in summary_result[0]:
                summary = summary_result[0]['summary_text'].strip()
                logging.info(f"Generated summary ({len(summary.split())} words): {summary[:150]}...")
                return summary
            else: logging.warning(f"Summarization unexpected format: {summary_result}"); return text
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            if "memory" in str(e).lower(): logging.warning("OOM during summarization."); self.release_summarizer_memory(); release_memory()
            return text

    def count_tokens(self, text):
        if self.tokenizer:
            try: return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception as e: logging.warning(f"CLIP tokenizer failed: {e}. Falling back to word count."); return len(text.split())
        else: return len(text.split())

    def _extract_quality_style_terms(self, text):
        found_terms = {}; text_lower = text.lower()
        sorted_terms = sorted(self.quality_style_terms.keys(), key=len, reverse=True)
        processed_indices = set()
        for term in sorted_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            for match in re.finditer(pattern, text_lower):
                start, end = match.span()
                if not any(max(start, p_start) < min(end, p_end) for p_start, p_end in processed_indices):
                    found_terms[term] = self.quality_style_terms[term]; processed_indices.add((start, end))
        return sorted(found_terms.items(), key=lambda item: item[1], reverse=True)

    def _extract_elements_spacy_v7(self, doc):
        elements = defaultdict(lambda: {"attributes": set(), "relationships": [], "base_weight": 1.0, "is_core_subject": False, "mentions": 0, "context": "foreground"})
        core_subject_tokens = set(); all_noun_chunks = {}
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass", "dobj", "attr") and token.pos_ in ("NOUN", "PROPN"):
                 if token.head.pos_ == "VERB" and token.head.lemma_ not in self.ignore_elements: core_subject_tokens.add(token)
            elif token.dep_ == "ROOT" and token.pos_ in ("NOUN", "PROPN"): core_subject_tokens.add(token)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip(); chunk_text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', chunk_text); chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
            if not chunk_text or chunk_text in self.ignore_elements or len(chunk_text) <= 1: continue
            all_noun_chunks[chunk_text] = chunk; is_core = any(core_token in chunk for core_token in core_subject_tokens)
            elements[chunk_text]["mentions"] += 1
            if is_core: elements[chunk_text]["is_core_subject"] = True; elements[chunk_text]["base_weight"] = max(elements[chunk_text]["base_weight"], 1.15)
        for chunk_text, chunk in all_noun_chunks.items():
            element_data = elements[chunk_text]; is_background_related = False
            for token in chunk:
                token_lemma, token_text = token.lemma_.lower(), token.text.lower()
                if token_lemma in self.background_keywords: is_background_related = True
                if token.dep_ == "amod" and token.head in chunk and token_lemma not in self.ignore_elements:
                    element_data["attributes"].add(token_lemma);
                    if token_lemma in self.color_terms: element_data["attributes"].add(f"color_{token_lemma}")
                elif token.dep_ == "nummod" and token.head in chunk:
                     if token_text.isdigit() or (token_lemma.isdigit() and token_lemma not in self.ignore_elements): element_data["attributes"].add(token_text)
                elif token.dep_ == "compound" and token.head in chunk:
                    if token_lemma not in self.ignore_elements and token != chunk.root: element_data["attributes"].add(token_lemma)
            if is_background_related: element_data["context"] = "background"
            element_data["base_weight"] = min(MAX_PROMPT_WEIGHT - 0.1, element_data["base_weight"] + math.log1p(element_data["mentions"] -1) * 0.05)
        element_names = set(elements.keys())
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN", "VERB"):
                source_chunk_text = self._find_element_chunk_for_token(token.head, all_noun_chunks)
                if source_chunk_text and source_chunk_text in element_names:
                    for child in token.children:
                        if child.dep_ == "pobj":
                            target_chunk_text = self._find_element_chunk_for_token(child, all_noun_chunks)
                            if target_chunk_text and target_chunk_text in element_names and target_chunk_text != source_chunk_text:
                                relation = (token.lemma_.lower(), target_chunk_text)
                                if relation not in elements[source_chunk_text]["relationships"]:
                                    elements[source_chunk_text]["relationships"].append(relation)
                                    elements[source_chunk_text]["base_weight"] = min(MAX_PROMPT_WEIGHT - 0.1, elements[source_chunk_text]["base_weight"] + 0.02)
                                    elements[target_chunk_text]["base_weight"] = min(MAX_PROMPT_WEIGHT - 0.1, elements[target_chunk_text]["base_weight"] + 0.02)
            elif token.pos_ == "VERB" and token.lemma_ not in self.ignore_elements:
                 subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]; objects = [c for c in token.children if c.dep_ in ("dobj", "attr", "pobj")]
                 for subj_token in subjects:
                     source_chunk_text = self._find_element_chunk_for_token(subj_token, all_noun_chunks)
                     if source_chunk_text and source_chunk_text in element_names:
                         for obj_token in objects:
                             target_chunk_text = self._find_element_chunk_for_token(obj_token, all_noun_chunks)
                             if target_chunk_text and target_chunk_text in element_names and target_chunk_text != source_chunk_text:
                                 relation = (token.lemma_.lower(), target_chunk_text)
                                 if relation not in elements[source_chunk_text]["relationships"]:
                                     elements[source_chunk_text]["relationships"].append(relation)
                                     elements[source_chunk_text]["base_weight"] = min(MAX_PROMPT_WEIGHT - 0.1, elements[source_chunk_text]["base_weight"] + 0.02)
                                     elements[target_chunk_text]["base_weight"] = min(MAX_PROMPT_WEIGHT - 0.1, elements[target_chunk_text]["base_weight"] + 0.02)
        final_elements = {}
        for name, data in elements.items():
            if name in self.ignore_elements and not data["is_core_subject"]: continue
            cleaned_attrs = {attr for attr in data["attributes"] if attr not in self.ignore_elements and len(attr) > 1}
            data["attributes"] = sorted(list(cleaned_attrs))
            valid_relationships = []
            for rel_type, target_name in data["relationships"]:
                 if target_name in elements and not (target_name in self.ignore_elements and not elements[target_name]["is_core_subject"]): valid_relationships.append((rel_type, target_name))
            data["relationships"] = valid_relationships
            if name: final_elements[name] = data
        return final_elements

    def _find_element_chunk_for_token(self, token, all_noun_chunks):
        for chunk_text, chunk_obj in all_noun_chunks.items():
            if token in chunk_obj: return chunk_text
        if token.pos_ in ("NOUN", "PROPN"):
             token_text = token.text.lower().strip(); token_text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', token_text); token_text = re.sub(r'\s+', ' ', token_text).strip()
             if token_text in all_noun_chunks: return token_text
        return None

    def _enhance_with_semantics(self, elements, doc_or_text):
        self.semantic_analyzer.initialize()
        if not self.semantic_analyzer.initialized:
            logging.warning("Semantic analyzer NA. Using base weights.")
            for name, data in elements.items(): data["importance_score"] = 0.5; data["final_weight"] = max(MIN_PROMPT_WEIGHT, min(data["base_weight"], MAX_PROMPT_WEIGHT))
            return elements
        if isinstance(doc_or_text, str): full_text = doc_or_text; doc = nlp(full_text)
        elif isinstance(doc_or_text, spacy.tokens.Doc): doc = doc_or_text; full_text = doc.text
        else: logging.error(f"Invalid input type: {type(doc_or_text)}."); return elements # Fallback handled in caller
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()] or [full_text]
        for name, data in elements.items():
            element_parts = set(name.split()); context_sents = [s for s in sentences if any(part in s.lower() for part in element_parts)] or sentences
            importance = self.semantic_analyzer.analyze_element_importance(name, context_sents, full_text); data["importance_score"] = importance
            base_weight = data["base_weight"]; context_mod = 0.95 if data["context"] == "background" else 1.0
            importance_multiplier = 0.8 + (importance * 0.4)
            final_weight = base_weight * context_mod * importance_multiplier
            data["final_weight"] = max(MIN_PROMPT_WEIGHT, min(final_weight, MAX_PROMPT_WEIGHT))
        return elements

    def _create_element_phrase(self, name, data):
        attributes = data.get("attributes", set())
        colors = sorted([a.split('_')[1] for a in attributes if a.startswith("color_")])
        numbers = sorted([a for a in attributes if a.isdigit()])
        other_attrs = sorted([a for a in attributes if not a.startswith("color_") and not a.isdigit()])
        phrase_parts = numbers[:1] + colors[:1] + other_attrs[:2] + [name]
        unique_parts = []; seen_parts = set()
        for part in phrase_parts:
            if part and part not in seen_parts: unique_parts.append(part); seen_parts.add(part)
        phrase = " ".join(unique_parts).strip(); return re.sub(r'\s+', ' ', phrase)

    def _format_weighted_phrase(self, text, weight, remove_weights=False):
        clean_text = re.sub(r'\s+', ' ', text).strip();
        if not clean_text: return ""
        if remove_weights: return clean_text
        clamped_weight = max(MIN_PROMPT_WEIGHT, min(weight, MAX_PROMPT_WEIGHT))
        if 0.98 < clamped_weight < 1.02: return clean_text
        else: weight_str = f"{clamped_weight:.2f}".rstrip('0').rstrip('.'); return f"({clean_text}:{weight_str})"

    def optimize_prompt_v7(self, original_prompt, remove_prompt_weights=False):
        logging.info(f"Optimizing prompt (v7 logic) | Remove Weights: {remove_prompt_weights}...")
        analysis_output = {"original_prompt": original_prompt, "remove_weights_flag": remove_prompt_weights}
        processed_prompt = original_prompt; original_word_count = len(original_prompt.split()); summary_generated = False
        if original_word_count > self.summarization_trigger_length:
            logging.info(f"Prompt length ({original_word_count}) > trigger ({self.summarization_trigger_length}). Summarizing...")
            summary = self._summarize_prompt(original_prompt)
            if summary and isinstance(summary, str) and summary.strip() and summary != original_prompt:
                processed_prompt = summary; analysis_output["summary_generated"] = True; analysis_output["summary_text"] = summary; summary_generated = True
                logging.info("Using summary for analysis.")
            else: logging.warning("Summarization failed or same as original. Using original."); analysis_output["summary_generated"] = False; analysis_output["summary_text"] = None
        else: analysis_output["summary_generated"] = False; analysis_output["summary_text"] = None

        logging.info(f"Analyzing {'summary' if summary_generated else 'original prompt'}...")
        doc = nlp(processed_prompt); elements = self._extract_elements_spacy_v7(doc)
        elements = self._enhance_with_semantics(elements, doc); analysis_output["elements_extracted_count"] = len(elements)
        quality_terms = self._extract_quality_style_terms(original_prompt); analysis_output["quality_terms_extracted"] = quality_terms

        prompt_parts = []; current_tokens = 0; added_elements = set()
        num_quality_terms_to_add = min(len(quality_terms), 5); quality_terms_added_log = []
        for term, weight in quality_terms[:num_quality_terms_to_add]:
            formatted_term = self._format_weighted_phrase(term, weight, remove_weights=remove_prompt_weights)
            term_tokens = self.count_tokens(formatted_term)
            if current_tokens + term_tokens <= self.token_limit: prompt_parts.append(formatted_term); current_tokens += term_tokens; quality_terms_added_log.append((term, weight))
            else: logging.debug(f"Skipping quality term '{term}' (limit)."); break
        analysis_output["quality_terms_used_in_prompt"] = quality_terms_added_log

        core_elements = sorted([(name, data) for name, data in elements.items() if data["is_core_subject"]], key=lambda x: -x[1]["final_weight"])
        logging.debug(f"Found {len(core_elements)} core elements.")
        for name, data in core_elements:
            if name in added_elements: continue
            phrase = self._create_element_phrase(name, data); formatted_phrase = self._format_weighted_phrase(phrase, data["final_weight"], remove_weights=remove_prompt_weights)
            phrase_tokens = self.count_tokens(formatted_phrase)
            if current_tokens + phrase_tokens <= self.token_limit: prompt_parts.append(formatted_phrase); current_tokens += phrase_tokens; added_elements.add(name)
            else:
                unweighted_phrase = phrase; unweighted_tokens = self.count_tokens(unweighted_phrase)
                if current_tokens + unweighted_tokens <= self.token_limit: logging.debug(f"Adding core '{name}' unweighted (limit)."); prompt_parts.append(unweighted_phrase); current_tokens += unweighted_tokens; added_elements.add(name)
                else: logging.debug(f"Skipping core '{name}' (limit).")

        fg_elements = sorted([(name, data) for name, data in elements.items() if not data["is_core_subject"] and data["context"] == "foreground"], key=lambda x: -x[1]["final_weight"])
        logging.debug(f"Found {len(fg_elements)} foreground elements.")
        for name, data in fg_elements:
            if name in added_elements: continue
            phrase = self._create_element_phrase(name, data); formatted_phrase = self._format_weighted_phrase(phrase, data["final_weight"], remove_weights=remove_prompt_weights)
            phrase_tokens = self.count_tokens(formatted_phrase)
            if current_tokens + phrase_tokens <= self.token_limit: prompt_parts.append(formatted_phrase); current_tokens += phrase_tokens; added_elements.add(name)
            else: logging.debug(f"Skipping foreground '{name}' (limit).")

        relationships_to_add = []; added_relationships_log = []
        for source_name in added_elements:
            source_data = elements.get(source_name);
            if not source_data: continue
            for rel_type, target_name in source_data["relationships"]:
                 if target_name in added_elements:
                     target_data = elements.get(target_name);
                     if not target_data: continue
                     rel_text = f"{source_name} {rel_type} {target_name}"
                     rel_weight = (source_data["final_weight"] + target_data["final_weight"]) / 2.0 * 0.95
                     relationships_to_add.append((rel_text, rel_weight))
        relationships_to_add.sort(key=lambda x: -x[1]); logging.debug(f"Found {len(relationships_to_add)} potential relationships.")
        for rel_text, rel_weight in relationships_to_add:
            formatted_rel = self._format_weighted_phrase(rel_text, rel_weight, remove_weights=remove_prompt_weights)
            rel_tokens = self.count_tokens(formatted_rel)
            if current_tokens + rel_tokens <= self.token_limit: prompt_parts.append(formatted_rel); current_tokens += rel_tokens; added_relationships_log.append(rel_text)
            else: logging.debug(f"Skipping relationship '{rel_text}' (limit)."); break
        analysis_output["relationships_used_in_prompt"] = added_relationships_log

        bg_elements = sorted([(name, data) for name, data in elements.items() if data["context"] == "background" and name not in added_elements], key=lambda x: -x[1]["final_weight"])
        logging.debug(f"Found {len(bg_elements)} background elements.")
        for name, data in bg_elements:
            phrase = self._create_element_phrase(name, data); bg_weight = data["final_weight"] * 0.9
            formatted_phrase = self._format_weighted_phrase(phrase, bg_weight, remove_weights=remove_prompt_weights)
            phrase_tokens = self.count_tokens(formatted_phrase)
            if current_tokens + phrase_tokens <= self.token_limit: prompt_parts.append(formatted_phrase); current_tokens += phrase_tokens; added_elements.add(name)
            else: logging.debug(f"Skipping background '{name}' (limit).")

        final_prompt = ", ".join(filter(None, prompt_parts)).strip(); final_prompt = re.sub(r'\s*,\s*', ', ', final_prompt); final_prompt = re.sub(r'(,\s*)+', ', ', final_prompt); final_prompt = re.sub(r'^,\s*|\s*,$', '', final_prompt).strip()
        if not final_prompt:
            logging.warning("Optimization resulted in empty prompt. Using original (maybe truncated).")
            if self.count_tokens(original_prompt) > self.token_limit:
                 if self.tokenizer: tokens = self.tokenizer.encode(original_prompt, add_special_tokens=False); truncated_tokens = tokens[:self.token_limit]; final_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True).strip(); logging.warning(f"Original truncated: '{final_prompt}'")
                 else: final_prompt = " ".join(original_prompt.split()[:self.token_limit]); logging.warning(f"Original roughly truncated: '{final_prompt}'")
            else: final_prompt = original_prompt
            final_token_count = self.count_tokens(final_prompt)
        else: final_token_count = self.count_tokens(final_prompt)
        logging.info(f"Optimized prompt ({final_token_count} tokens): {final_prompt}")

        analysis_output["elements_included_in_prompt_details"] = { name: {"phrase": self._create_element_phrase(name, data), "attributes": list(data["attributes"]), "relationships": data["relationships"], "base_weight": round(data["base_weight"], 3), "importance_score": round(data.get("importance_score", -1.0), 3), "final_weight": round(data["final_weight"], 3), "is_core": data["is_core_subject"], "context": data["context"], "mentions": data["mentions"]} for name, data in elements.items() if name in added_elements }
        analysis_output["elements_included_names_only"] = list(added_elements)
        return {"prompt": final_prompt, "token_count": final_token_count, "analysis": analysis_output}

    def release_summarizer_memory(self):
        if self.summarizer or self.summarizer_tokenizer:
            summarizer_ref, tokenizer_ref = self.summarizer, self.summarizer_tokenizer
            model_ref = getattr(self.summarizer, 'model', None)
            self.summarizer, self.summarizer_tokenizer, self.summarizer_initialized = None, None, False
            release_memory(summarizer_ref, tokenizer_ref, model_ref); logging.info("Summarizer resources released.")

    def release_memory(self):
        self.semantic_analyzer.release_memory(); self.release_summarizer_memory()
        logging.info("DynamicPromptOptimizer resources released.")


# --- Image Enhancement (Unchanged from v7) ---
def enhance_image_quality(image, enhancement_level=DEFAULT_ENHANCE_LEVEL):
    if not isinstance(image, Image.Image) or not enhancement_level or enhancement_level <= 0.05: return image
    try:
        img = image.copy(); level = max(0.0, min(1.0, enhancement_level))
        contrast_factor = 1.0 + (0.08 * level); enhancer = ImageEnhance.Contrast(img); img = enhancer.enhance(contrast_factor)
        color_factor = 1.0 + (0.06 * level); enhancer = ImageEnhance.Color(img); img = enhancer.enhance(color_factor)
        sharpness_radius = 0.6 + (0.9 * level); sharpness_percent = 60 + int(60 * level); sharpness_threshold = 3
        img = img.filter(ImageFilter.UnsharpMask(radius=sharpness_radius, percent=sharpness_percent, threshold=sharpness_threshold))
        logging.info(f"Applied image enhancement with level {level:.2f}"); return img
    except Exception as e: logging.error(f"Image enhancement failed: {e}"); return image


# --- Main Pipeline Class ---

class EnhancedSDXLPipeline:
    """
    Enhanced SDXL pipeline (v8) integrating prompt optimization, dual semantics,
    CLIP scoring (normalized, original prompt), optional weight removal,
    guidance randomization, increased candidates, LCM refinement,
    and CLIP Interrogator hybrid scoring.
    """
    def __init__(self, base_model_path=SDXL_BASE_MODEL, refiner_model_path=SDXL_REFINER_MODEL, device=DEVICE):
        self.device = device
        self.base_model_path = base_model_path
        self.refiner_model_path = refiner_model_path # This might be LCM or standard refiner path
        self.is_lcm_refiner = (refiner_model_path == LCM_REFINER_MODEL and LCM_AVAILABLE) # Track if we are using LCM
        self.txt2img_pipe = None
        self.refiner_pipe = None # Will hold either LCM or Standard refiner

        if self.device.type == "cuda": self.torch_dtype, self.variant = torch.float16, "fp16"
        elif self.device.type == "mps": self.torch_dtype, self.variant = torch.float32, None
        else: self.torch_dtype, self.variant = torch.float32, None
        logging.info(f"Using {self.device} with {self.torch_dtype} dtype.")

        self._load_sdxl_models()
        self.prompt_optimizer = DynamicPromptOptimizer(device=self.device, summarizer_model=T5_SUMMARIZER_MODEL_NAME)
        self.clip_scorer = CLIPScorer(device=self.device)
        # --- v8: Initialize CLIP Interrogator (lazy loaded on first use) ---
        # get_clip_interrogator(self.device) # Optional: Pre-initialize here

    def _load_sdxl_models(self):
        """Loads SDXL base and the appropriate refiner (LCM or Standard)."""
        logging.info(f"Loading SDXL models (Base: {self.base_model_path}, Refiner: {self.refiner_model_path})...")
        load_success = False
        for use_safetensors_option in [True, False]:
            if load_success: break
            logging.info(f"Attempting load with use_safetensors={use_safetensors_option}...")
            try:
                logging.info("Loading base model...")
                self.txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
                    self.base_model_path, torch_dtype=self.torch_dtype,
                    variant=self.variant, use_safetensors=use_safetensors_option,
                ).to(self.device)
                logging.info("Base model loaded.")

                # --- v8 Change: Load LCM or Standard Refiner ---
                if self.refiner_model_path:
                    if self.is_lcm_refiner:
                        # Use helper to load LCM, sharing components
                        self.refiner_pipe = load_lcm_refiner(
                            base_pipe=self.txt2img_pipe,
                            device=self.device,
                            dtype=self.torch_dtype,
                            variant=self.variant
                        )
                        if self.refiner_pipe is None:
                            logging.error("Failed to load LCM Refiner. Refinement stage will be skipped.")
                            self.is_lcm_refiner = False # Mark as failed
                    else:
                        # Load standard SDXL Img2Img Refiner
                        logging.info("Loading standard SDXL Refiner...")
                        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                            self.refiner_model_path,
                            text_encoder_2=self.txt2img_pipe.text_encoder_2,
                            vae=self.txt2img_pipe.vae,
                            torch_dtype=self.torch_dtype,
                            variant=self.variant,
                            use_safetensors=use_safetensors_option,
                        ).to(self.device)
                        logging.info("Standard SDXL Refiner loaded.")
                else:
                    self.refiner_pipe = None
                    logging.info("Refiner model path not specified, skipping refiner loading.")

                # Model optimizations
                try:
                    self.txt2img_pipe.enable_model_cpu_offload()
                    if self.refiner_pipe: self.refiner_pipe.enable_model_cpu_offload()
                    logging.info("Enabled model CPU offloading.")
                except AttributeError:
                     logging.info("CPU offloading not available/enabled. Using attention slicing.")
                     self.txt2img_pipe.enable_attention_slicing()
                     if self.refiner_pipe: self.refiner_pipe.enable_attention_slicing()

                load_success = True
                logging.info(f"Models loaded successfully (use_safetensors={use_safetensors_option}).")

            except Exception as e:
                logging.warning(f"Model loading failed (safetensors={use_safetensors_option}): {e}")
                self._release_sdxl_memory(); release_memory()

        if not load_success:
            raise RuntimeError("Could not load required SDXL models.")

    def _run_inference_stage(self, pipe_type, prompt, negative_prompt, generator,
                           num_inference_steps, guidance_scale, **kwargs):
        """Runs inference stage (base or refiner), handling LCM specifics."""
        pipe = self.txt2img_pipe if pipe_type == "txt2img" else self.refiner_pipe
        stage_name = "Base Generation" if pipe_type == "txt2img" else "Refinement (LCM)" if self.is_lcm_refiner and pipe_type == "refiner" else "Refinement (Standard)"

        if pipe is None:
            logging.warning(f"Skipping {stage_name} stage: Pipeline not loaded.")
            return kwargs.get("image", None) if pipe_type == "refiner" else None

        logging.info(f"Starting {stage_name} | Steps: {num_inference_steps}, Guidance: {guidance_scale:.1f}")
        logging.debug(f"  {stage_name} Prompt: {prompt[:150]}...")
        logging.debug(f"  {stage_name} Neg Prompt: {negative_prompt[:150]}...")

        try:
            if not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
                seed = generator.initial_seed() if isinstance(generator, torch.Generator) else random.randint(0, 2**32 - 1)
                logging.warning(f"Recreating generator on CPU for {stage_name} with seed {seed}.")
                generator = torch.Generator(device='cpu').manual_seed(seed)

            params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "generator": generator,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "output_type": "pil",
            }
            params.update(kwargs)

            # --- v8 Change: Adjust parameters based on pipe type ---
            if pipe_type == "txt2img":
                params.pop("image", None); params.pop("strength", None)
            elif pipe_type == "refiner":
                params.pop("width", None); params.pop("height", None)
                # LCM might implicitly handle strength via steps, but we pass it.
                # Ensure 'strength' is present if it's expected by the refiner pipe.
                if "strength" not in params and "strength" in pipe.config.keys(): # Check if strength is expected
                     logging.warning(f"Strength parameter missing for {stage_name}, using default 0.3")
                     params["strength"] = 0.3 # Default if missing, adjust as needed

            # Run inference
            with torch.no_grad():
                result = pipe(**params)

            image = result.images[0] if result.images else None
            if image is None: raise RuntimeError("Inference returned no images.")

            logging.info(f"{stage_name} stage completed successfully.")
            return image

        except Exception as e:
            logging.error(f"{stage_name} stage failed: {e}", exc_info=True)
            if "memory" in str(e).lower():
                logging.error("OOM error during inference. Try reducing resolution/candidates.")
                release_memory(); self._release_sdxl_memory()
            return kwargs.get("image", None) if pipe_type == "refiner" else None

    def _calculate_prompt_complexity(self, analysis_data):
        """Estimates prompt complexity (0-10 scale). Unchanged from v7."""
        if not analysis_data or not isinstance(analysis_data, dict): return 0.0
        elements_included = analysis_data.get("elements_included_in_prompt_details", {})
        if not elements_included: return 0.0
        num_core = sum(1 for d in elements_included.values() if d.get("is_core"))
        num_fg = sum(1 for d in elements_included.values() if not d.get("is_core") and d.get("context") == "foreground")
        num_bg = sum(1 for d in elements_included.values() if d.get("context") == "background")
        total_attrs = sum(len(d.get("attributes", [])) for d in elements_included.values())
        total_rels = len(analysis_data.get("relationships_used_in_prompt", []))
        complexity_score = (num_core*2.5) + (num_fg*1.5) + (num_bg*0.5) + (total_attrs*0.15) + (total_rels*0.8)
        if analysis_data.get("summary_generated"): complexity_score += 2.5
        scaled_complexity = min(10.0, max(0.0, complexity_score / 3.5))
        return round(scaled_complexity, 1)

    def generate(self,
                 prompt: str,
                 negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
                 num_inference_steps: int = BASE_STEPS,
                 seed: Optional[int] = None,
                 output_path: Optional[str] = None,
                 width: int = 1024,
                 height: int = 1024,
                 guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
                 refiner_strength: float = DEFAULT_REFINER_STRENGTH,
                 # --- v8 Changes: Refiner steps config & Interrogation flag ---
                 lcm_refiner_steps: int = DEFAULT_LCM_REFINER_STEPS, # Steps for LCM refiner
                 refiner_steps_ratio: float = DEFAULT_STD_REFINER_STEPS_RATIO, # Ratio for standard refiner
                 enable_clip_interrogation: bool = False, # Flag to enable hybrid scoring
                 clip_interrogation_mode: str = 'fast', # 'fast' or 'best'
                 # --- End v8 Changes ---
                 enhance_level: float = DEFAULT_ENHANCE_LEVEL,
                 num_candidates: int = DEFAULT_NUM_CANDIDATES,
                 clip_score_prompt_override: Optional[str] = None,
                 remove_prompt_weights: bool = False,
                 randomize_guidance: bool = False
                 ):
        """
        Main generation function (v8) with LCM, hybrid CLIP scoring, and other enhancements.
        """
        start_time = time.time()
        results = {
            "parameters": {k:v for k, v in locals().items() if k != 'self'},
            "output_dir": None, "master_seed": None, "candidate_seeds": [],
            "optimization_details": {}, "generation_prompt_used": None, "prompt_complexity": 0.0,
            "adjusted_base_steps": None, "clip_scoring_prompts": [],
            "all_candidate_results": [], "best_candidate_index": -1,
            # --- v8 Change: Store hybrid score ---
            "best_hybrid_clip_score": -0.1, # Use 0-1 range
            "best_original_clip_score": -0.1, # Also store original avg score of best candidate
            # --- End v8 Change ---
            "selected_seed": None, "selected_candidate_details": {}, "final_image_path": None,
            "total_processing_time_seconds": 0, "final_image": None, "error": None,
            "using_lcm_refiner": self.is_lcm_refiner, # Info field
            "clip_interrogation_enabled": enable_clip_interrogation and CLIP_INTERROGATOR_AVAILABLE, # Info field
        }

        # --- Setup Output Directory ---
        run_output_dir = None
        if output_path:
            try:
                prompt_safe = re.sub(r'[^\w\-]+', '_', prompt[:40]).strip('_')
                seed_str = str(seed) if seed is not None else 'rand'
                ts = time.strftime("%Y%m%d_%H%M%S")
                run_output_dir = os.path.join(output_path, f"sdxl_v8_{prompt_safe}_{seed_str}_{ts}")
                os.makedirs(run_output_dir, exist_ok=True)
                results["output_dir"] = run_output_dir
                logging.info(f"Output will be saved to: {run_output_dir}")
            except OSError as e: logging.warning(f"Cannot create output dir '{output_path}': {e}. Saving disabled.")

        # --- Seed Management ---
        master_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        results["master_seed"] = master_seed
        candidate_seeds = [master_seed + i for i in range(num_candidates)]
        results["candidate_seeds"] = candidate_seeds
        logging.info(f"Master seed: {master_seed} | Generating {num_candidates} candidates (seeds: {candidate_seeds[0]}...).")

        # --- Prompt Optimization (v7 logic) ---
        logging.info("Optimizing prompt...")
        optimized_data = {}
        generation_prompt = prompt
        try:
            self.prompt_optimizer.semantic_analyzer.initialize()
            optimized_data = self.prompt_optimizer.optimize_prompt_v7(prompt, remove_prompt_weights=remove_prompt_weights)
            generation_prompt = optimized_data["prompt"]
            results["optimization_details"] = optimized_data.get("analysis", {})
            results["generation_prompt_used"] = generation_prompt
            logging.info("Prompt optimization successful.")
        except Exception as e:
            logging.error(f"Prompt optimization failed: {e}. Using original prompt.", exc_info=True)
            # Fallback logic (simplified from v7, assuming optimizer handles truncation on failure)
            generation_prompt = results["optimization_details"].get("prompt_used_for_generation", prompt)
            results["generation_prompt_used"] = generation_prompt

        # --- Calculate Complexity & Adjust Steps ---
        complexity = self._calculate_prompt_complexity(results["optimization_details"])
        step_adjustment = int(complexity * (COMPLEXITY_STEP_BOOST / 10.0))
        adjusted_base_steps = max(20, min(80, num_inference_steps + step_adjustment))
        results["prompt_complexity"] = complexity
        results["adjusted_base_steps"] = adjusted_base_steps
        logging.info(f"Complexity: {complexity:.1f}/10 | Adjusted base steps: {adjusted_base_steps}")

        # --- Determine CLIP Scoring Prompts (v7 logic) ---
        if clip_score_prompt_override: primary_clip_prompt = clip_score_prompt_override
        else: primary_clip_prompt = prompt
        scoring_prompts_list = [primary_clip_prompt]
        summary_text = results["optimization_details"].get("summary_text")
        if summary_text and summary_text != prompt: scoring_prompts_list.append(summary_text)
        scoring_prompts_list.append(f"high quality professional photo of {primary_clip_prompt}")
        if generation_prompt != primary_clip_prompt: scoring_prompts_list.append(generation_prompt)
        seen_prompts = set(); unique_scoring_prompts = []
        for p in scoring_prompts_list:
            p_strip = p.strip()
            if p_strip and p_strip not in seen_prompts: unique_scoring_prompts.append(p_strip); seen_prompts.add(p_strip)
        results["clip_scoring_prompts"] = unique_scoring_prompts
        logging.info(f"Prepared {len(unique_scoring_prompts)} unique prompts for original CLIP scoring.")

        # --- Generate Candidates ---
        logging.info(f"Generating {num_candidates} candidates...")
        candidate_results = []

        for i in tqdm(range(num_candidates), desc="Generating Candidates", unit="candidate"):
            candidate_seed = candidate_seeds[i]
            generator = torch.Generator(device='cpu').manual_seed(candidate_seed)
            candidate_data = {"candidate_index": i, "seed": candidate_seed, "stages": {}, "error": None, "final_image": None}
            current_image = None; stage_times = {}

            current_guidance_scale = guidance_scale
            if randomize_guidance:
                current_guidance_scale = round(random.uniform(GUIDANCE_RANDOM_RANGE[0], GUIDANCE_RANDOM_RANGE[1]), 1)
            candidate_data["guidance_scale_used"] = current_guidance_scale

            try:
                # === Stage 1: Base Generation ===
                s1_start = time.time()
                current_image = self._run_inference_stage("txt2img", generation_prompt, negative_prompt, generator, adjusted_base_steps, current_guidance_scale, width=width, height=height)
                stage_times["base"] = time.time() - s1_start
                if current_image is None: raise RuntimeError("Base generation failed.")
                base_img_path = self._save_image(current_image, run_output_dir, f"cand_{i+1}_s{candidate_seed}_1_base_g{current_guidance_scale:.1f}.png")
                candidate_data["stages"]["base"] = {"success": True, "time_sec": stage_times["base"], "output_path": base_img_path}

                # === Stage 2: Refinement (LCM or Standard) ===
                refined_image = current_image
                if self.refiner_pipe and refiner_strength > 0.01:
                    s2_start = time.time()
                    # --- v8 Change: Determine refiner steps ---
                    if self.is_lcm_refiner:
                        refiner_steps = max(1, lcm_refiner_steps) # Use specific LCM steps
                        refiner_type = "LCM"
                    else: # Standard refiner
                        refiner_steps = max(5, int(adjusted_base_steps * refiner_steps_ratio)) # Use ratio
                        refiner_type = "Standard"
                    logging.debug(f"Running {refiner_type} refiner for cand {i+1} | Steps: {refiner_steps}, Strength: {refiner_strength:.2f}")

                    refiner_generator = torch.Generator(device='cpu').manual_seed(candidate_seed + 1000)
                    refined_image = self._run_inference_stage(
                        pipe_type="refiner",
                        prompt=generation_prompt, negative_prompt=negative_prompt,
                        image=current_image, strength=refiner_strength,
                        generator=refiner_generator, num_inference_steps=refiner_steps,
                        guidance_scale=current_guidance_scale # LCM uses guidance too
                    )
                    stage_times["refiner"] = time.time() - s2_start

                    if refined_image is None:
                        logging.warning(f"Refinement (Stage 2 - {refiner_type}) failed for candidate {i+1}. Using base image.")
                        refined_image = current_image # Fallback
                        candidate_data["stages"]["refiner"] = {"success": False, "time_sec": stage_times.get("refiner", 0), "warning": "Refinement failed."}
                    else:
                        current_image = refined_image
                        refined_img_path = self._save_image(current_image, run_output_dir, f"cand_{i+1}_s{candidate_seed}_2_refined_{refiner_type.lower()}.png")
                        candidate_data["stages"]["refiner"] = {"success": True, "time_sec": stage_times["refiner"], "type": refiner_type, "steps": refiner_steps, "strength": refiner_strength, "output_path": refined_img_path}
                else: candidate_data["stages"]["refiner"] = {"success": False, "skipped": True}

                # === Stage 3: Enhancement (Optional) ===
                enhanced_image = current_image
                if enhance_level > 0.05:
                    s3_start = time.time()
                    enhanced_image = enhance_image_quality(current_image, enhance_level)
                    stage_times["enhancement"] = time.time() - s3_start
                    current_image = enhanced_image
                    enhanced_img_path = self._save_image(current_image, run_output_dir, f"cand_{i+1}_s{candidate_seed}_3_enhanced.png")
                    candidate_data["stages"]["enhancement"] = {"success": True, "time_sec": stage_times["enhancement"], "level": enhance_level, "output_path": enhanced_img_path}
                else: candidate_data["stages"]["enhancement"] = {"success": False, "skipped": True}

                candidate_data["final_image"] = current_image
                candidate_data["total_time_sec"] = sum(stage_times.values())
                logging.info(f"Candidate {i+1} finished in {candidate_data['total_time_sec']:.2f}s.")

            except Exception as e:
                 logging.error(f"Error processing candidate {i+1} (seed {candidate_seed}): {e}", exc_info=True)
                 candidate_data["error"] = str(e); candidate_data["final_image"] = None

            candidate_results.append(candidate_data)
            release_memory(current_image, refined_image, enhanced_image); release_memory()

        results["all_candidate_results"] = candidate_results

        # --- Candidate Selection using CLIP Score (Original + Optional Hybrid) ---
        best_index = -1
        best_hybrid_score = -0.1 # Use hybrid score for selection if enabled
        best_original_score_at_best_hybrid = -0.1 # Track original score of the best hybrid candidate

        valid_candidates = [(idx, c) for idx, c in enumerate(candidate_results)
                            if c.get("final_image") is not None and c.get("error") is None]

        if not valid_candidates:
             logging.error("No valid image candidates were generated successfully.")
             results["error"] = "All candidates failed during generation stages."
        else:
            logging.info(f"Running scoring on {len(valid_candidates)} valid candidates...")
            self.clip_scorer.initialize() # Ensure CLIP model is ready

            if not self.clip_scorer.initialized:
                logging.error("CLIP scorer failed to initialize. Cannot score candidates.")
                # Fallback: Select first valid candidate
                best_index = valid_candidates[0][0]
                logging.warning(f"Selecting first valid candidate (Index: {best_index}) due to CLIP scorer failure.")
                # Assign neutral scores
                results["all_candidate_results"][best_index]["original_clip_score"] = 0.0
                results["all_candidate_results"][best_index]["clip_details"] = {"error": "CLIP scorer init failed"}
                if results["clip_interrogation_enabled"]:
                    results["all_candidate_results"][best_index]["reverse_clip_score"] = 0.0
                    results["all_candidate_results"][best_index]["hybrid_clip_score"] = 0.0
            else:
                # --- Score each valid candidate ---
                for idx, cand_data in tqdm(valid_candidates, desc="Scoring Candidates", unit="candidate"):
                    cand_image = cand_data["final_image"]
                    original_avg_score = 0.0
                    reverse_caption = None
                    reverse_score = 0.0
                    hybrid_score = 0.0

                    try:
                        # 1. Calculate Original Multi-Prompt CLIP Score
                        original_avg_score, score_details = self.clip_scorer.calculate_multi_prompt_score(
                            cand_image, unique_scoring_prompts
                        )
                        results["all_candidate_results"][idx]["original_clip_score"] = round(original_avg_score, 5)
                        results["all_candidate_results"][idx]["clip_details"] = {k: round(v, 5) for k, v in score_details.items()}
                        logging.debug(f"Cand {idx+1} Orig CLIP: {original_avg_score:.4f}")

                        # 2. Run CLIP Interrogation & Reverse Score (if enabled)
                        if results["clip_interrogation_enabled"]:
                            interrogation_start = time.time()
                            reverse_caption = run_clip_interrogation(cand_image, mode=clip_interrogation_mode)
                            interrogation_time = time.time() - interrogation_start
                            results["all_candidate_results"][idx]["clip_interrogation_caption"] = reverse_caption
                            results["all_candidate_results"][idx]["clip_interrogation_time_sec"] = round(interrogation_time, 2)

                            if reverse_caption:
                                # Calculate score between image and its reverse caption
                                reverse_score = self.clip_scorer.calculate_clip_score(cand_image, reverse_caption)
                                results["all_candidate_results"][idx]["reverse_clip_score"] = round(reverse_score, 5)
                                logging.debug(f"Cand {idx+1} Reverse CLIP: {reverse_score:.4f}")

                                # Calculate Hybrid Score
                                hybrid_score = (original_avg_score + reverse_score) / 2.0
                                results["all_candidate_results"][idx]["hybrid_clip_score"] = round(hybrid_score, 5)
                                logging.debug(f"Cand {idx+1} Hybrid CLIP: {hybrid_score:.4f}")
                            else:
                                # Interrogation failed or returned empty, hybrid score defaults to original
                                hybrid_score = original_avg_score # Fallback
                                results["all_candidate_results"][idx]["reverse_clip_score"] = None
                                results["all_candidate_results"][idx]["hybrid_clip_score"] = round(hybrid_score, 5)
                                logging.debug(f"Cand {idx+1} Interrogation failed, Hybrid CLIP = Orig CLIP: {hybrid_score:.4f}")
                        else:
                            # Interrogation disabled, hybrid score is just the original score
                            hybrid_score = original_avg_score
                            results["all_candidate_results"][idx]["hybrid_clip_score"] = round(hybrid_score, 5) # Store for consistency

                        # 3. Update Best Candidate Tracking (using Hybrid Score if enabled, else Original)
                        current_comparison_score = hybrid_score if results["clip_interrogation_enabled"] else original_avg_score

                        if current_comparison_score > best_hybrid_score: # Use best_hybrid_score tracker for comparison
                            best_hybrid_score = current_comparison_score
                            best_index = idx
                            # Store the original score corresponding to this best candidate
                            best_original_score_at_best_hybrid = original_avg_score
                        elif best_index == -1: # Handle first valid score
                             best_hybrid_score = current_comparison_score
                             best_index = idx
                             best_original_score_at_best_hybrid = original_avg_score

                    except Exception as e:
                        logging.error(f"Scoring error for candidate {idx+1}: {e}", exc_info=False)
                        results["all_candidate_results"][idx]["original_clip_score"] = 0.0
                        results["all_candidate_results"][idx]["clip_details"] = {"error": f"Scoring failed: {str(e)}"}
                        if results["clip_interrogation_enabled"]:
                            results["all_candidate_results"][idx]["reverse_clip_score"] = 0.0
                            results["all_candidate_results"][idx]["hybrid_clip_score"] = 0.0

                # --- Report Best Candidate ---
                results["best_hybrid_clip_score"] = round(best_hybrid_score, 5)
                results["best_original_clip_score"] = round(best_original_score_at_best_hybrid, 5) # Store the original score of the winner
                results["best_candidate_index"] = best_index

                if best_index != -1:
                     selected_seed = results["all_candidate_results"][best_index]["seed"]
                     score_type = "Hybrid" if results["clip_interrogation_enabled"] else "Original Avg"
                     logging.info(f"Best candidate: Index {best_index + 1} (Seed: {selected_seed}) with {score_type} CLIP score: {best_hybrid_score:.4f}")
                     if results["clip_interrogation_enabled"]:
                         logging.info(f"  (Original Avg Score: {best_original_score_at_best_hybrid:.4f}, Reverse Score: {results['all_candidate_results'][best_index].get('reverse_clip_score', 'N/A'):.4f})")
                else:
                     logging.warning("Scoring completed, but no candidate selected as best.")

                # --- Release CLIP Scorer Memory ---
                self.clip_scorer.release_memory()
                # --- Release Interrogator Memory (if used) ---
                if results["clip_interrogation_enabled"]:
                    release_clip_interrogator_memory()

        # --- Final Image Selection and Saving ---
        final_image = None
        if best_index != -1:
            best_cand_data = results["all_candidate_results"][best_index]
            final_image = best_cand_data.get("final_image")
            results["final_image"] = final_image
            results["selected_seed"] = best_cand_data["seed"]
            results["selected_candidate_details"] = { k: v for k, v in best_cand_data.items() if k != "final_image" }

            if run_output_dir and final_image:
                # Use hybrid score in filename if available, else original
                score_to_use = best_hybrid_score if results["clip_interrogation_enabled"] and best_hybrid_score > 0 else best_original_score_at_best_hybrid
                score_str = f"hclip_{score_to_use:.4f}".replace(".", "_") if results["clip_interrogation_enabled"] else f"oclip_{score_to_use:.4f}".replace(".", "_")
                fname = f"FINAL_cand_{best_index+1}_s{best_cand_data['seed']}_{score_str}.png"
                results["final_image_path"] = self._save_image(final_image, run_output_dir, fname)
                logging.info(f"Best image saved to: {results['final_image_path']}")
            elif final_image: logging.warning("Output directory not set. Final image not saved.")
            else: logging.warning("Best candidate selected, but final image data missing.")
        else:
             results["error"] = results.get("error", "No successful candidate found or selected.")
             logging.error(f"Generation failed: {results['error']}")

        # --- Post-Processing and Cleanup ---
        total_time = time.time() - start_time
        results["total_processing_time_seconds"] = round(total_time, 2)
        logging.info(f"Total processing time: {total_time:.2f} seconds.")

        # Clean up final report data
        for cand in results["all_candidate_results"]: cand.pop("final_image", None)
        if run_output_dir: self._save_analysis_json(run_output_dir, results)

        # Optional: remove final image from returned dict
        # results.pop("final_image", None)

        return results


    def _save_image(self, image: Optional[Image.Image], path: Optional[str], filename: str) -> Optional[str]:
        """Saves a PIL image."""
        if not path or not isinstance(image, Image.Image): return None
        try:
            safe_name = re.sub(r'[^\w\.\-]+', '_', filename)
            if not safe_name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")): safe_name += ".png"
            full_path = os.path.join(path, safe_name)
            image.save(full_path, quality=95)
            logging.debug(f"Image saved: {full_path}")
            return full_path
        except Exception as e: logging.warning(f"Failed to save image '{filename}': {e}"); return None

    def _save_analysis_json(self, path: str, results_data: dict):
        """Saves the results dictionary as a JSON report."""
        if not path: return
        try:
            filepath = os.path.join(path, "generation_report_v8.json")
            data_to_save = {}
            for key, value in results_data.items():
                if key == "final_image": continue
                try: json.dumps({key: value}); data_to_save[key] = value # Test serializability
                except TypeError: data_to_save[key] = json.loads(json.dumps({key: value}, default=lambda o: f"<non-serializable: {type(o).__name__}>"))[key]
                except Exception as json_err: data_to_save[key] = f"<serialization error: {str(json_err)}>"
            with open(filepath, "w", encoding='utf-8') as f: json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            logging.info(f"Analysis report saved to: {filepath}")
        except Exception as e: logging.error(f"Failed to save JSON report: {e}", exc_info=True)

    def release(self):
        """Releases all loaded models and clears memory."""
        logging.info("Releasing all pipeline resources...")
        self._release_sdxl_memory()
        if hasattr(self, 'clip_scorer') and self.clip_scorer: self.clip_scorer.release_memory(); self.clip_scorer = None
        if hasattr(self, 'prompt_optimizer') and self.prompt_optimizer: self.prompt_optimizer.release_memory(); self.prompt_optimizer = None
        # --- v8 Change: Release interrogator ---
        release_clip_interrogator_memory()
        release_memory() # Global release
        logging.info("Pipeline resources released.")

    def _release_sdxl_memory(self):
        """Specifically releases the SDXL Diffusers pipelines."""
        logging.debug("Releasing SDXL model memory...")
        pipe1, pipe2 = self.txt2img_pipe, self.refiner_pipe
        self.txt2img_pipe, self.refiner_pipe = None, None
        release_memory(pipe1, pipe2)
        logging.debug("SDXL model memory released.")


# --- System Wrapper ---

class AdvancedSDXLSystem:
    """High-level wrapper for the EnhancedSDXLPipeline (v8)."""
    def __init__(self, base_model_path=SDXL_BASE_MODEL, refiner_model_path=SDXL_REFINER_MODEL):
        self.base_model_path = base_model_path
        self.refiner_model_path = refiner_model_path
        self.pipeline = None
        # --- v8 Change: Updated defaults ---
        self.default_settings = {
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            "num_inference_steps": BASE_STEPS,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "width": 1024, "height": 1024,
            "refiner_strength": DEFAULT_REFINER_STRENGTH,
            "lcm_refiner_steps": DEFAULT_LCM_REFINER_STEPS, # Added
            "refiner_steps_ratio": DEFAULT_STD_REFINER_STEPS_RATIO, # Kept for standard refiner fallback
            "enable_clip_interrogation": False, # Added, default off
            "clip_interrogation_mode": 'fast', # Added
            "enhance_level": DEFAULT_ENHANCE_LEVEL,
            "num_candidates": DEFAULT_NUM_CANDIDATES,
            "clip_score_prompt_override": None,
            "remove_prompt_weights": False,
            "randomize_guidance": False,
        }
        logging.info("AdvancedSDXLSystem (v8) initialized.")

    def _initialize_pipeline(self):
        """Initializes the underlying EnhancedSDXLPipeline (v8)."""
        if self.pipeline is None:
            logging.info("Initializing EnhancedSDXLPipeline (v8)...")
            try:
                self.pipeline = EnhancedSDXLPipeline(
                    base_model_path=self.base_model_path,
                    refiner_model_path=self.refiner_model_path, # Will be LCM or standard path
                    device=DEVICE
                )
                logging.info("EnhancedSDXLPipeline (v8) initialized successfully.")
            except Exception as e:
                logging.error(f"Pipeline initialization failed: {e}", exc_info=True)
                self.pipeline = None
                raise RuntimeError(f"Failed to initialize pipeline: {e}")

    def generate(self, prompt: str, output_path: Optional[str] = None, seed: Optional[int] = None, **kwargs):
        """Generates an image using the v8 pipeline."""
        logging.info(f"Received generation request: '{prompt[:70]}...'")
        try:
            self._initialize_pipeline()
            if self.pipeline is None: raise RuntimeError("Pipeline not initialized.")

            settings = self.default_settings.copy()
            valid_keys = set(settings.keys())
            overrides = {}
            for k, v in kwargs.items():
                if k in valid_keys:
                    if v is not None: settings[k] = v; overrides[k] = v
                    else: logging.debug(f"Ignoring None for setting '{k}'.")
                else: logging.warning(f"Ignoring unknown setting '{k}'.")
            if overrides: logging.info(f"Applying overrides: {overrides}")
            logging.debug(f"Final generation settings: {settings}")

            result = self.pipeline.generate(prompt=prompt, output_path=output_path, seed=seed, **settings)
            return result

        except Exception as e:
            logging.critical(f"Generation process failed critically: {e}", exc_info=True)
            return {"error": f"Critical failure: {str(e)}", "final_image": None, "final_image_path": None, "parameters": {"prompt": prompt, "seed": seed, **kwargs}}

    def release_resources(self):
        """Releases the pipeline and resources."""
        if self.pipeline:
            logging.info("Releasing resources via AdvancedSDXLSystem...")
            self.pipeline.release()
            self.pipeline = None
        release_memory() # Global release
        logging.info("AdvancedSDXLSystem resources released.")

# --- Example Usage & Command Line Interface ---

def run_generation_example(prompt, output_dir="sdxl_output_v8", seed=None, **kwargs):
    """Example function demonstrating the AdvancedSDXLSystem (v8)."""
    system = None
    try:
        # Determine refiner path based on --no_refiner flag
        use_refiner = not kwargs.pop("no_refiner", False)
        # Use LCM path if available and refiner enabled, else standard path, else None
        if use_refiner:
            refiner_path = LCM_REFINER_MODEL if LCM_AVAILABLE else "stabilityai/stable-diffusion-xl-refiner-1.0"
            logging.info(f"Refiner enabled ({'LCM' if LCM_AVAILABLE else 'Standard'}). Path: {refiner_path}")
        else:
            refiner_path = None
            logging.info("Refiner disabled by user.")

        system = AdvancedSDXLSystem(
            base_model_path=SDXL_BASE_MODEL,
            refiner_model_path=refiner_path
        )

        logging.info("--- Starting Generation Process (v8) ---")
        result = system.generate(prompt=prompt, output_path=output_dir, seed=seed, **kwargs)
        logging.info("--- Generation Process Finished ---")

        if result.get("error"): logging.error(f"Generation failed: {result['error']}")
        elif result.get("final_image_path") or result.get("final_image"):
            logging.info("Generation completed successfully!")
            if result.get("final_image_path"): logging.info(f"Best image saved to: {result['final_image_path']}")
            else: logging.warning("Image generated but not saved (check output_dir).")

            # Report best score (Hybrid if enabled, else Original)
            score_type = "Hybrid" if result.get("clip_interrogation_enabled") else "Original Avg"
            best_score = result.get("best_hybrid_clip_score", -0.1) if result.get("clip_interrogation_enabled") else result.get("best_original_clip_score", -0.1)
            best_cand_idx = result.get("best_candidate_index", -1)
            cand_num_str = f"Candidate {best_cand_idx + 1}" if best_cand_idx != -1 else "N/A"

            if best_score >= 0.0:
                logging.info(f"Best {score_type} CLIP score (0-1): {best_score:.4f} ({cand_num_str})")
                # Show breakdown if hybrid
                if result.get("clip_interrogation_enabled"):
                     orig_score = result.get("best_original_clip_score", -0.1)
                     rev_score = result.get("selected_candidate_details", {}).get("reverse_clip_score", None)
                     rev_score_str = f"{rev_score:.4f}" if rev_score is not None else "N/A"
                     logging.info(f"  (Breakdown: Original Avg={orig_score:.4f}, Reverse={rev_score_str})")
            else: logging.warning("CLIP score not available or selection failed.")

            if result.get("total_processing_time_seconds"): logging.info(f"Total time: {result['total_processing_time_seconds']:.2f}s")

            # Optimization Summary (same as v7)
            if result.get("optimization_details"):
                logging.info("--- Prompt Optimization Summary ---")
                opt = result["optimization_details"]
                logging.info(f"  Gen Prompt ({result.get('generation_prompt_used', 'N/A').count(' ')+1} words): {result.get('generation_prompt_used', 'N/A')[:200]}...")
                logging.info(f"  Summary: {opt.get('summary_generated', False)} | Weights Removed: {opt.get('remove_weights_flag', False)}")
                logging.info(f"  Quality Terms: {len(opt.get('quality_terms_used_in_prompt', []))} | Elements: {len(opt.get('elements_included_names_only', []))} | Relations: {len(opt.get('relationships_used_in_prompt', []))}")
                logging.info("---------------------------------")
        else: logging.error("Generation finished, but no final image or error reported.")
        return result
    except Exception as e:
        logging.critical(f"Unexpected critical error in runner: {e}", exc_info=True)
        return {"error": f"Critical runner error: {str(e)}"}
    finally:
        if system: system.release_resources()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced SDXL Pipeline (v8 - LCM & Hybrid Score)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("prompt", nargs="?", type=str, default=None, help="Text prompt. Uses a default complex example if omitted.")
    parser.add_argument("--output_dir", type=str, default="sdxl_output_v8", help="Output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Random if None.")

    # --- Generation Parameter Overrides ---
    parser.add_argument("--neg", type=str, default=argparse.SUPPRESS, dest="negative_prompt", help=f"Override negative prompt.")
    parser.add_argument("--steps", type=int, default=argparse.SUPPRESS, dest="num_inference_steps", help=f"Base inference steps (default: {BASE_STEPS}). Adjusted by complexity.")
    parser.add_argument("--guidance", "--cfg", type=float, default=argparse.SUPPRESS, dest="guidance_scale", help=f"Guidance scale (CFG) (default: {DEFAULT_GUIDANCE_SCALE:.1f}).")
    parser.add_argument("--width", type=int, default=argparse.SUPPRESS, help="Image width (default: 1024).")
    parser.add_argument("--height", type=int, default=argparse.SUPPRESS, help="Image height (default: 1024).")
    parser.add_argument("--refiner_strength", type=float, default=argparse.SUPPRESS, help=f"Refiner strength (0-1) (default: {DEFAULT_REFINER_STRENGTH:.2f}).")
    # --- v8 Change: Specific LCM steps ---
    parser.add_argument("--lcm_refiner_steps", type=int, default=argparse.SUPPRESS, help=f"Number of steps for LCM refiner (if used) (default: {DEFAULT_LCM_REFINER_STEPS}).")
    parser.add_argument("--refiner_ratio", type=float, default=argparse.SUPPRESS, dest="refiner_steps_ratio", help=f"Ratio of base steps for *standard* refiner (fallback) (default: {DEFAULT_STD_REFINER_STEPS_RATIO:.2f}).")
    parser.add_argument("--enhance", type=float, default=argparse.SUPPRESS, dest="enhance_level", help=f"Post-processing enhancement level (0-1) (default: {DEFAULT_ENHANCE_LEVEL:.2f}).")
    parser.add_argument("--candidates", type=int, default=argparse.SUPPRESS, dest="num_candidates", help=f"Number of candidates (default: {DEFAULT_NUM_CANDIDATES}).")

    # --- v7/v8 Specific Flags ---
    parser.add_argument("--clip_prompt", type=str, default=argparse.SUPPRESS, dest="clip_score_prompt_override", help="Override prompt for CLIP scoring (defaults to original).")
    parser.add_argument("--no_prompt_weights", action="store_true", help="Generate prompt without (word:weight) syntax.")
    parser.add_argument("--randomize_guidance", action="store_true", help=f"Randomize guidance scale per candidate {GUIDANCE_RANDOM_RANGE}.")
    parser.add_argument("--no_refiner", action="store_true", help="Completely disable the refiner stage.")
    # --- v8 Change: Interrogation Flags ---
    parser.add_argument("--enable_clip_interrogation", action="store_true", help="Enable CLIP Interrogator for reverse captioning and hybrid scoring.")
    parser.add_argument("--clip_interrogation_mode", type=str, default='fast', choices=['fast', 'best'], help="Mode for CLIP Interrogator ('fast' or 'best').")


    args = parser.parse_args()

    # Default prompt (same as v7)
    default_prompt = """A modern cozy bedroom with a Scandinavian design aesthetic, featuring a warm, inviting atmosphere. The main wall is painted a soft teal blue, with one side wall in light gray and a darker gray accent wall.
A red fabric sofa bed is placed in the center, made up neatly with white bedding and a cream-colored throw blanket.
Behind the bed, there is a tall white open bookshelf filled with books (mainly in red, orange, and neutral colors), small decor items, and a potted plant at the top.
A sleek white floor lamp with a minimalistic conical shade stands next to the bookshelf.
A black modern clock with white hands hangs on the teal wall, along with a cluster of framed photographs and minimalist art pieces arranged in a gallery style.
To the right, a wicker chair with a curved back and metal legs is placed near the window, topped with a striped cushion in earthy tones (red, brown, beige).
In front of the chair, there is a small footstool with a turquoise cushion and black thin metal legs.
A square wooden coffee table with a lattice design sits at the center, holding a small potted plant, a couple of books, and light decor items.
A woven round side table with a wooden top is placed beside the bed, holding a water bottle and books.
Full-height white curtains cover large windows, letting in abundant natural sunlight, casting soft shadows.
The flooring is dark wood with a beige rug underneath the bed and table area.
The overall mood is bright, fresh, minimalistic, and cozy."""
    prompt_to_use = args.prompt if args.prompt else default_prompt
    if not args.prompt: logging.info(f"Using default prompt: '{default_prompt[:100]}...'")

    # Prepare kwargs, filtering out SUPPRESS and handled args
    gen_kwargs = vars(args).copy()
    handled_args = ["prompt", "output_dir", "seed"] # Handled by runner directly
    final_gen_kwargs = {}
    for k, v in gen_kwargs.items():
        if k in handled_args: continue
        if v is not argparse.SUPPRESS:
            final_gen_kwargs[k] = v
        # Handle boolean flags explicitly
        elif k in ["no_prompt_weights", "randomize_guidance", "no_refiner", "enable_clip_interrogation"]:
             if gen_kwargs.get(k) == True: final_gen_kwargs[k] = True
             # else: default False is handled by the generate function

    # Run the example
    run_generation_example(
        prompt=prompt_to_use,
        output_dir=args.output_dir,
        seed=args.seed,
        **final_gen_kwargs
    )

    logging.info("Script finished.")

