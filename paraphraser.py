#!/usr/bin/env python3
"""
Optimized Indonesian Paraphraser System
Enhanced accuracy with IndoT5 + Smart Synonym Integration
"""

import json
import os
import random
import re
import torch
import math
from typing import List, Dict, Optional, Union, Tuple
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ===================== ENHANCED HYBRID PARAPHRASER =====================
class EnhancedHybridParaphraser:
    def __init__(self, synonym_dict_path: str):
        self.synonym_dict = self.load_synonym_dict(synonym_dict_path)
        self.transformation_history = []
        self.pos_patterns = self.load_pos_patterns()
        self.punctuation_chars = '.,!?;:"\'()[]{}«»'
        
    def load_synonym_dict(self, path: str) -> Dict:
        """Load synonym dictionary with enhanced format support"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Support both old and new formats
            if data and isinstance(list(data.values())[0], dict):
                if 'sinonim' in list(data.values())[0]:
                    return data
                else:
                    # Convert old format
                    converted = {}
                    for word, synonyms in data.items():
                        if isinstance(synonyms, list):
                            converted[word] = {"sinonim": synonyms, "tag": "unknown"}
                        elif isinstance(synonyms, dict):
                            converted[word] = synonyms
                    return converted
            return {}
        except Exception as e:
            print(f"Error loading synonym dictionary: {e}")
            return {}
    
    def load_pos_patterns(self) -> Dict:
        """Load part-of-speech patterns for better transformations"""
        return {
            'verb_patterns': [
                r'\b(me\w+)\b',  # me- prefix verbs
                r'\b(ber\w+)\b',  # ber- prefix verbs
                r'\b(ter\w+)\b',  # ter- prefix verbs
                r'\b(di\w+)\b',   # di- prefix verbs (passive)
            ],
            'noun_patterns': [
                r'\b(pe\w+an)\b',  # pe-an nouns
                r'\b(ke\w+an)\b',  # ke-an nouns
                r'\b(\w+isme)\b',  # -isme nouns
                r'\b(\w+itas)\b',  # -itas nouns
            ],
            'adjective_patterns': [
                r'\b(ter\w+)\b',   # ter- adjectives
                r'\b(\w+if)\b',    # -if adjectives
                r'\b(\w+al)\b',    # -al adjectives
            ]
        }
    
    def get_word_context(self, word: str, text: str, position: int) -> Dict:
        """Get context information for better synonym selection"""
        words = text.split()
        context = {
            'prev_word': words[position-1] if position > 0 else None,
            'next_word': words[position+1] if position < len(words)-1 else None,
            'sentence_pos': 'start' if position == 0 else 'middle' if position < len(words)-1 else 'end'
        }
        return context
    
    def score_synonym(self, original: str, synonym: str, context: Dict) -> float:
        """Score synonym based on context and linguistic rules"""
        score = 1.0
        
        # Length similarity preference
        len_ratio = len(synonym) / len(original) if len(original) > 0 else 1
        if 0.5 <= len_ratio <= 2.0:
            score += 0.2
        
        # Context-based scoring
        if context.get('prev_word'):
            # Avoid awkward combinations
            prev_word = context['prev_word'].lower()
            if prev_word in ['sangat', 'lebih', 'paling'] and len(synonym.split()) > 1:
                score -= 0.3
        
        # Formal vs informal preference
        formal_indicators = ['yang', 'dapat', 'akan', 'telah', 'adalah']
        informal_indicators = ['ya', 'sih', 'nih', 'dong']
        
        if any(indicator in synonym.lower() for indicator in formal_indicators):
            score += 0.1
        if any(indicator in synonym.lower() for indicator in informal_indicators):
            score -= 0.1
        
        return max(score, 0.1)
    
    def smart_synonym_selection(self, word: str, text: str, position: int) -> str:
        """Select best synonym based on context and scoring"""
        synonyms = self.get_synonyms(word)
        if not synonyms:
            return word
        
        context = self.get_word_context(word, text, position)
        
        # Score each synonym
        scored_synonyms = []
        for synonym in synonyms:
            score = self.score_synonym(word, synonym, context)
            scored_synonyms.append((synonym, score))
        
        # Sort by score and add randomness
        scored_synonyms.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 3 with weighted randomness
        top_synonyms = scored_synonyms[:3]
        if len(top_synonyms) == 1:
            return top_synonyms[0][0]
        
        # Weighted random selection
        weights = [s[1] for s in top_synonyms]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(synonyms)
        
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for synonym, weight in top_synonyms:
            current_weight += weight
            if rand_val <= current_weight:
                return synonym
        
        return top_synonyms[0][0]
    
    def preserve_word_format(self, original: str, replacement: str) -> str:
        """Enhanced word format preservation"""
        punctuation = ''
        clean_original = original
        
        # Extract punctuation
        punct_chars = '.,!?;:"\'()[]{}«»""'''
        while clean_original and clean_original[-1] in punct_chars:
            punctuation = clean_original[-1] + punctuation
            clean_original = clean_original[:-1]
        
        # Apply capitalization
        if clean_original.isupper():
            replacement = replacement.upper()
        elif clean_original.istitle():
            replacement = replacement.capitalize()
        elif clean_original.islower():
            replacement = replacement.lower()
        
        return replacement + punctuation
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms with filtering"""
        clean_word = word.lower().strip(self.punctuation_chars)
        if clean_word in self.synonym_dict:
            synonyms = self.synonym_dict[clean_word].get('sinonim', [])
            # Filter out very long synonyms or inappropriate ones
            filtered = [s for s in synonyms if len(s.split()) <= 3 and s != clean_word]
            return filtered
        return []
    
    def enhanced_synonym_replacement(self, text: str, replacement_rate: float = 0.4, 
                                   preserve_keywords: bool = True) -> Tuple[str, List[int], List[Dict]]:
        """Enhanced synonym replacement with context awareness"""
        words = text.split()
        result = []
        replaced_positions = []
        replacement_details = []
        
        # Important keywords to preserve
        preserve_words = {
            'tidak', 'bukan', 'jangan', 'harus', 'wajib', 'dapat', 'bisa',
            'akan', 'telah', 'sudah', 'sedang', 'adalah', 'ialah', 'yaitu',
            'ini', 'itu', 'tersebut', 'yang', 'dengan', 'untuk', 'dalam'
        }
        
        for i, word in enumerate(words):
            clean_word = word.lower().strip(self.punctuation_chars)
            
            # Skip preservation words
            if preserve_keywords and clean_word in preserve_words:
                result.append(word)
                continue
            
            # Skip very short words
            if len(clean_word) <= 2:
                result.append(word)
                continue
            
            synonyms = self.get_synonyms(clean_word)
            
            if synonyms and random.random() < replacement_rate:
                # Use smart selection
                chosen_synonym = self.smart_synonym_selection(clean_word, text, i)
                formatted_synonym = self.preserve_word_format(word, chosen_synonym)
                
                result.append(formatted_synonym)
                replaced_positions.append(i)
                replacement_details.append({
                    'original': word,
                    'replacement': formatted_synonym,
                    'position': i,
                    'available_synonyms': synonyms,
                    'selection_score': self.score_synonym(clean_word, chosen_synonym, 
                                                       self.get_word_context(clean_word, text, i))
                })
            else:
                result.append(word)
        
        return ' '.join(result), replaced_positions, replacement_details
    
    def enhanced_active_to_passive(self, sentence: str) -> str:
        """Enhanced active to passive transformation"""
        # More comprehensive patterns for Indonesian
        patterns = [
            {
                'pattern': r'(\w+)\s+(me\w+)\s+(\w+)',
                'transform': lambda m: f"{m.group(3)} di{m.group(2)[2:]} oleh {m.group(1)}"
            },
            {
                'pattern': r'(\w+)\s+(akan|telah|sudah|sedang)\s+(me\w+)\s+(\w+)',
                'transform': lambda m: f"{m.group(4)} {m.group(2)} di{m.group(3)[2:]} oleh {m.group(1)}"
            },
            {
                'pattern': r'(\w+)\s+(ber\w+)\s+(\w+)',
                'transform': lambda m: f"{m.group(3)} di{m.group(2)[3:]} oleh {m.group(1)}"
            }
        ]
        
        for p in patterns:
            match = re.search(p['pattern'], sentence, re.IGNORECASE)
            if match:
                try:
                    return p['transform'](match)
                except:
                    continue
        return sentence
    
    def enhanced_clause_reordering(self, text: str) -> str:
        """Enhanced clause reordering with better pattern recognition"""
        # Handle different types of compound sentences
        conjunctions = ['karena', 'ketika', 'jika', 'meskipun', 'walaupun', 'setelah', 'sebelum', 'agar', 'supaya']
        
        for conj in conjunctions:
            pattern = rf'({conj}[^,]+),\s*(.+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(2)}, {match.group(1)}"
        
        # Handle other comma-separated clauses
        if ',' in text:
            parts = text.split(',', 1)
            if len(parts) == 2:
                # Check if reordering makes sense
                first_part = parts[0].strip()
                second_part = parts[1].strip()
                
                # Don't reorder if first part is very short (likely not a clause)
                if len(first_part.split()) >= 3:
                    return f"{second_part}, {first_part}"
        
        return text
    
    def enhanced_conjunction_replacement(self, text: str) -> str:
        """Enhanced conjunction replacement with context awareness"""
        conjunctions = {
            'dan': ['serta', 'juga', 'beserta', 'plus', 'kemudian'],
            'atau': ['ataupun', 'maupun', 'bahkan'],
            'tetapi': ['namun', 'akan tetapi', 'tapi', 'namun demikian', 'kendati'],
            'karena': ['sebab', 'oleh karena', 'dikarenakan', 'akibat', 'lantaran'],
            'jika': ['apabila', 'bila', 'seandainya', 'kalau', 'manakala'],
            'meskipun': ['walaupun', 'sekalipun', 'kendati', 'biarpun', 'sungguhpun'],
            'sehingga': ['hingga', 'sampai', 'maka', 'alhasil'],
            'kemudian': ['lalu', 'selanjutnya', 'berikutnya', 'sesudah itu']
        }
        
        # Choose conjunction to replace based on frequency and context
        result = text
        for original, alternatives in conjunctions.items():
            if original in result.lower():
                # Context-aware selection
                if 'formal' in result.lower() or 'resmi' in result.lower():
                    # Prefer formal alternatives
                    formal_alts = [alt for alt in alternatives if len(alt.split()) >= 2]
                    if formal_alts:
                        replacement = random.choice(formal_alts)
                    else:
                        replacement = random.choice(alternatives)
                else:
                    replacement = random.choice(alternatives)
                
                result = re.sub(rf'\b{original}\b', replacement, result, flags=re.IGNORECASE)
                break
        
        return result
    
    def enhanced_modifier_adjustment(self, text: str) -> str:
        """Enhanced modifier addition/removal with semantic awareness"""
        # Intensity modifiers
        intensifiers = {
            'penting': ['sangat penting', 'amat penting', 'benar-benar penting'],
            'baik': ['lebih baik', 'sangat baik', 'amat baik'],
            'besar': ['cukup besar', 'sangat besar', 'amat besar'],
            'kecil': ['agak kecil', 'cukup kecil', 'relatif kecil'],
            'tinggi': ['cukup tinggi', 'sangat tinggi', 'amat tinggi'],
            'rendah': ['agak rendah', 'cukup rendah', 'relatif rendah'],
            'mudah': ['lebih mudah', 'sangat mudah', 'amat mudah'],
            'sulit': ['agak sulit', 'cukup sulit', 'sangat sulit']
        }
        
        # Removal patterns
        remove_patterns = {
            r'\bsangat\s+': '',
            r'\bamat\s+': '',
            r'\bcukup\s+': '',
            r'\blebih\s+': '',
            r'\bagak\s+': '',
            r'\brelatif\s+': '',
            r'\blumayan\s+': ''
        }
        
        if random.random() < 0.5:
            # Add intensifiers
            for word, intensified in intensifiers.items():
                if word in text and not any(mod in text for mod in ['sangat', 'amat', 'cukup', 'lebih']):
                    replacement = random.choice(intensified)
                    text = re.sub(rf'\b{word}\b', replacement, text, flags=re.IGNORECASE)
                    break
        else:
            # Remove intensifiers
            for pattern, replacement in remove_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                    break
        
        return text
    
    def calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity for transformation selection"""
        words = text.split()
        
        # Average word length
        avg_word_len = sum(len(word) for word in words) / len(words) if words else 0
        
        # Sentence length
        sentence_len = len(words)
        
        # Complex conjunctions
        complex_conjunctions = ['meskipun', 'walaupun', 'sekalipun', 'kendati', 'dikarenakan']
        complex_conj_count = sum(1 for conj in complex_conjunctions if conj in text.lower())
        
        # Calculate complexity score
        complexity = (avg_word_len / 10) + (sentence_len / 50) + (complex_conj_count * 0.2)
        return min(complexity, 1.0)
    
    def apply_contextual_transformations(self, text: str, variation_index: int) -> str:
        """Apply transformations based on text complexity and variation index"""
        complexity = self.calculate_text_complexity(text)
        
        # Define transformation sets based on complexity
        if complexity < 0.3:
            # Simple transformations for simple text
            transformation_sets = [
                [self.enhanced_conjunction_replacement],
                [self.enhanced_modifier_adjustment],
                [self.enhanced_conjunction_replacement, self.enhanced_modifier_adjustment]
            ]
        elif complexity < 0.6:
            # Medium transformations
            transformation_sets = [
                [self.enhanced_active_to_passive, self.enhanced_conjunction_replacement],
                [self.enhanced_clause_reordering, self.enhanced_modifier_adjustment],
                [self.enhanced_conjunction_replacement, self.enhanced_modifier_adjustment]
            ]
        else:
            # Complex transformations for complex text
            transformation_sets = [
                [self.enhanced_active_to_passive, self.enhanced_conjunction_replacement],
                [self.enhanced_clause_reordering, self.enhanced_modifier_adjustment],
                [self.enhanced_conjunction_replacement, self.enhanced_clause_reordering, self.enhanced_active_to_passive]
            ]
        
        transforms = transformation_sets[variation_index % len(transformation_sets)]
        result = text
        
        for transform in transforms:
            try:
                new_result = transform(result)
                if new_result != result:
                    self.transformation_history.append(transform.__name__)
                    result = new_result
            except Exception as e:
                print(f"Error in transformation {transform.__name__}: {e}")
                continue
        
        return result
    
    def enhanced_post_process(self, text: str) -> str:
        """Enhanced post-processing with better formatting"""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?;,])', r'\1', text)
        text = re.sub(r'([.!?;,])\s*', r'\1 ', text)
        
        # Fix spacing around quotes
        text = re.sub(r'\s*"\s*', '"', text)
        text = re.sub(r'\s*"\s*', '" ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Capitalize sentences properly
        sentences = re.split(r'([.!?]+\s*)', text)
        processed_sentences = []
        
        for i, sentence in enumerate(sentences):
            if i % 2 == 0:  # Actual sentence content
                sentence = sentence.strip()
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                processed_sentences.append(sentence)
            else:  # Punctuation
                processed_sentences.append(sentence)
        
        text = ''.join(processed_sentences)
        
        # Fix common issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'^[\s,]+|[\s,]+$', '', text)  # Leading/trailing spaces and commas
        
        return text.strip()
    
    def is_valid_paraphrase(self, original: str, paraphrase: str) -> bool:
        """Enhanced validation for paraphrases"""
        if not paraphrase or not paraphrase.strip():
            return False
        
        if original.lower().strip() == paraphrase.lower().strip():
            return False
        
        # Check length ratio
        len_ratio = len(paraphrase) / len(original) if len(original) else 0
        if len_ratio < 0.5 or len_ratio > 2.5:
            return False
        
        # Check keyword preservation
        original_keywords = set(self.extract_keywords(original))
        paraphrase_keywords = set(self.extract_keywords(paraphrase))
        
        if not original_keywords:
            return True
        
        # At least 30% of keywords should be preserved or replaced with synonyms
        preserved_or_replaced = 0
        for keyword in original_keywords:
            if keyword in paraphrase_keywords:
                preserved_or_replaced += 1
            else:
                # Check if replaced with synonym
                synonyms = self.get_synonyms(keyword)
                if any(syn in paraphrase_keywords for syn in synonyms):
                    preserved_or_replaced += 1
        
        preservation_ratio = preserved_or_replaced / len(original_keywords)
        
        return preservation_ratio >= 0.3
    
    def extract_keywords(self, text: str) -> List[str]:
        """Enhanced keyword extraction"""
        stop_words = {
            'yang', 'dan', 'atau', 'tetapi', 'karena', 'jika', 'untuk', 'dalam',
            'pada', 'dengan', 'dari', 'ke', 'oleh', 'ini', 'itu', 'adalah',
            'akan', 'telah', 'sudah', 'sedang', 'masih', 'juga', 'hanya',
            'dapat', 'bisa', 'harus', 'perlu', 'saja', 'pun', 'lah', 'kah',
            'nya', 'mu', 'ku', 'dia', 'ia', 'anda', 'saya', 'kami', 'kita',
            'mereka', 'beliau', 'bagaimana', 'mengapa', 'kapan', 'dimana',
            'siapa', 'apa', 'mana', 'berapa'
        }
        
        words = text.lower().split()
        keywords = []
        
        for word in words:
            clean_word = word.strip(self.punctuation_chars)
            if (clean_word not in stop_words and 
                len(clean_word) > 2 and 
                clean_word.isalpha()):
                keywords.append(clean_word)
        
        return keywords
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Enhanced similarity calculation"""
        # Jaccard similarity for words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard = len(intersection) / len(union) if union else 0
        
        # Character-level similarity
        chars1 = set(text1.lower().replace(' ', ''))
        chars2 = set(text2.lower().replace(' ', ''))
        
        char_intersection = chars1 & chars2
        char_union = chars1 | chars2
        
        char_similarity = len(char_intersection) / len(char_union) if char_union else 0
        
        # Combine both measures
        return (jaccard * 0.7) + (char_similarity * 0.3)
    
    def paraphrase(self, text: str, num_variations: int = 3, synonym_rate: float = 0.4, 
                  use_smart_replacement: bool = True) -> List[Dict]:
        """Enhanced paraphrase generation"""
        paraphrases = []
        max_attempts = num_variations * 3  # Allow multiple attempts
        
        for attempt in range(max_attempts):
            if len(paraphrases) >= num_variations:
                break
                
            self.transformation_history = []
            processed_text = ' '.join(text.split())
            
            # Vary synonym rate and approach
            current_rate = synonym_rate + (attempt * 0.05)
            current_rate = min(current_rate, 0.8)
            
            try:
                # Apply enhanced synonym replacement
                if use_smart_replacement:
                    text_with_synonyms, replaced_positions, replacement_details = self.enhanced_synonym_replacement(
                        processed_text, current_rate, preserve_keywords=True
                    )
                else:
                    text_with_synonyms, replaced_positions, replacement_details = self.enhanced_synonym_replacement(
                        processed_text, current_rate, preserve_keywords=False
                    )
                
                # Apply contextual transformations
                transformed_text = self.apply_contextual_transformations(text_with_synonyms, variation_index=attempt)
                
                # Enhanced post-processing
                final_text = self.enhanced_post_process(transformed_text)
                
                # Validate paraphrase
                if self.is_valid_paraphrase(text, final_text):
                    # Check for duplicates
                    if not any(existing['text'].lower() == final_text.lower() for existing in paraphrases):
                        paraphrases.append({
                            'text': final_text,
                            'transformations': self.transformation_history.copy(),
                            'similarity_score': self.calculate_similarity(text, final_text),
                            'replaced_positions': replaced_positions,
                            'replacement_details': replacement_details,
                            'synonym_rate_used': current_rate,
                            'complexity_score': self.calculate_text_complexity(final_text),
                            'attempt_number': attempt + 1
                        })
                
            except Exception as e:
                print(f"Error in paraphrase attempt {attempt + 1}: {e}")
                continue
        
        return paraphrases[:num_variations]

# ===================== ENHANCED INDOT5 PARAPHRASER =====================
class EnhancedIndoT5Paraphraser:
    def __init__(self, model_name: str = "LazarusNLP/IndoNanoT5-base", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
    def _get_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self) -> bool:
        """Load the T5 model with error handling"""
        try:
            print(f"Loading T5 model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            print(f"Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Failed to load T5 model: {e}")
            try:
                print("Trying alternative tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                
                self.model.to(self.device)
                self.model.eval()
                
                self.is_loaded = True
                print("Alternative tokenizer loaded successfully")
                return True
                
            except Exception as e2:
                print(f"Failed to load alternative tokenizer: {e2}")
                return False
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing for T5"""
        text = text.strip()
        text = ' '.join(text.split())
        
        # Add appropriate task prefix
        if "paraphrase" in self.model_name.lower():
            return f"paraphrase: {text}"
        elif "summarize" in self.model_name.lower():
            return f"summarize: {text}"
        else:
            return f"paraphrase: {text}"
    
    def postprocess_text(self, text: str) -> str:
        """Enhanced postprocessing for T5 output"""
        text = text.strip()
        
        # Remove task prefixes
        prefixes = [
            "paraphrased output:", "paraphrase:", "summarize:", "summary:",
            "hasil parafrase:", "ringkasan:", "terjemahan:"
        ]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Clean up common artifacts
        text = re.sub(r'^[:\-\s]*', '', text)
        text = re.sub(r'[:\-\s]*', '', text)
        
        # Capitalize first letter
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_paraphrases(self, text: str, num_variations: int = 3, 
                           generation_params: Optional[Dict] = None) -> List[str]:
        """Enhanced paraphrase generation with better parameters"""
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        # Enhanced default parameters
        default_params = {
            "max_length": min(len(text.split()) * 2, 128),
            "min_length": max(len(text.split()) // 2, 10),
            "num_beams": 6,
            "num_return_sequences": num_variations * 2,  # Generate more to filter
            "early_stopping": True,
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.3,
            "length_penalty": 1.0,
            "diversity_penalty": 0.5,
            "num_beam_groups": 2
        }
        
        if generation_params:
            default_params.update(generation_params)
        
        try:
            processed_text = self.preprocess_text(text)
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                processed_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate multiple times with different parameters for diversity
            all_paraphrases = []
            
            for temp in [0.7, 0.8, 0.9]:
                current_params = default_params.copy()
                current_params['temperature'] = temp
                current_params['num_return_sequences'] = num_variations
                
                with torch.no_grad():
                    outputs = self.model.generate(inputs, **current_params)
                
                # Decode outputs
                for output in outputs:
                    decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                    processed = self.postprocess_text(decoded)
                    
                    if (processed and 
                        processed.lower() != text.lower() and
                        len(processed.split()) >= 3):
                        all_paraphrases.append(processed)
            
            # Remove duplicates while preserving order
            unique_paraphrases = []
            seen = set()
            
            for para in all_paraphrases:
                para_lower = para.lower()
                if para_lower not in seen:
                    unique_paraphrases.append(para)
                    seen.add(para_lower)
            
            return unique_paraphrases[:num_variations]
            
        except Exception as e:
            print(f"Error in generate_paraphrases: {e}")
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Enhanced similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_fluency_score(self, text: str) -> float:
        """Calculate fluency score based on linguistic features"""
        words = text.split()
        
        # Length score
        length_score = min(len(words) / 20, 1.0)
        
        # Repetition penalty
        word_counts = Counter(words)
        repetition_penalty = 1.0 - (sum(1 for count in word_counts.values() if count > 1) / len(words))
        
        # Grammar indicators (simple heuristics)
        grammar_score = 1.0
        if text.count('yang yang') > 0:
            grammar_score -= 0.2
        if text.count('di di') > 0:
            grammar_score -= 0.2
        if text.count('untuk untuk') > 0:
            grammar_score -= 0.2
        
        return (length_score * 0.3 + repetition_penalty * 0.4 + grammar_score * 0.3)
    
    def paraphrase(self, text: str, num_variations: int = 3, 
                  generation_params: Optional[Dict] = None) -> List[Dict]:
        """Enhanced paraphrase generation with quality scoring"""
        paraphrases = self.generate_paraphrases(text, num_variations, generation_params)
        results = []
        
        for i, paraphrase in enumerate(paraphrases):
            similarity_score = self.calculate_similarity(text, paraphrase)
            fluency_score = self.calculate_fluency_score(paraphrase)
            
            # Quality score combines multiple factors
            quality_score = (fluency_score * 0.6) + ((1 - similarity_score) * 0.4)
            
            result = {
                'text': paraphrase,
                'method': 'IndoT5',
                'model': self.model_name,
                'similarity_score': similarity_score,
                'fluency_score': fluency_score,
                'quality_score': quality_score,
                'confidence': max(0.1, quality_score),
                'generation_rank': i + 1
            }
            results.append(result)
        
        # Sort by quality score
        results.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return results

# ===================== INTEGRATED PARAPHRASER =====================
class IntegratedParaphraser:
    def __init__(self, synonym_dict_path: str = 'sinonim.json', 
                 t5_model_name: str = "LazarusNLP/IndoNanoT5-base", 
                 enable_hybrid: bool = True, enable_t5: bool = True):
        self.enable_hybrid = enable_hybrid
        self.enable_t5 = enable_t5
        
        # Initialize enhanced hybrid paraphraser
        self.hybrid_paraphraser = None
        if enable_hybrid:
            if os.path.exists(synonym_dict_path):
                try:
                    self.hybrid_paraphraser = EnhancedHybridParaphraser(synonym_dict_path)
                    print(f"Enhanced hybrid paraphraser loaded with {len(self.hybrid_paraphraser.synonym_dict)} synonyms")
                except Exception as e:
                    print(f"Failed to load enhanced hybrid paraphraser: {e}")
                    self.enable_hybrid = False
            else:
                print(f"Synonym dictionary not found: {synonym_dict_path}")
                self.enable_hybrid = False
        
        # Initialize enhanced T5 paraphraser
        self.t5_paraphraser = None
        if enable_t5:
            try:
                self.t5_paraphraser = EnhancedIndoT5Paraphraser(t5_model_name)
                print(f"Enhanced T5 paraphraser initialized with model: {t5_model_name}")
            except Exception as e:
                print(f"Failed to initialize enhanced T5 paraphraser: {e}")
                self.enable_t5 = False
        
        self.methods_available = {
            'hybrid': self.enable_hybrid,
            't5': self.enable_t5
        }
        
        print(f"Integrated paraphraser initialized - Hybrid: {self.enable_hybrid}, T5: {self.enable_t5}")
    
    def paraphrase_hybrid(self, text: str, num_variations: int = 3, 
                         synonym_rate: float = 0.4, use_smart_replacement: bool = True,
                         preserve_keywords: bool = True) -> List[Dict]:
        """Generate paraphrases using enhanced hybrid method"""
        if not self.enable_hybrid or not self.hybrid_paraphraser:
            return []
        
        try:
            results = self.hybrid_paraphraser.paraphrase(
                text, num_variations, synonym_rate, use_smart_replacement
            )
            
            # Add method metadata
            for result in results:
                result['method'] = 'Enhanced Hybrid'
                result['model'] = 'Rule-based + Smart Synonyms'
                if 'confidence' not in result:
                    # Calculate confidence based on multiple factors
                    confidence = (
                        result.get('similarity_score', 0.5) * 0.3 +
                        (1 - result.get('similarity_score', 0.5)) * 0.4 +
                        result.get('complexity_score', 0.5) * 0.3
                    )
                    result['confidence'] = min(max(confidence, 0.1), 1.0)
            
            return results
            
        except Exception as e:
            print(f"Error in enhanced hybrid paraphrasing: {e}")
            return []
    
    def paraphrase_t5(self, text: str, num_variations: int = 3, 
                     generation_params: Optional[Dict] = None) -> List[Dict]:
        """Generate paraphrases using enhanced T5 method"""
        if not self.enable_t5 or not self.t5_paraphraser:
            return []
        
        try:
            return self.t5_paraphraser.paraphrase(text, num_variations, generation_params)
        except Exception as e:
            print(f"Error in enhanced T5 paraphrasing: {e}")
            return []
    
    def combine_and_rank_results(self, results: List[Dict], num_variations: int) -> List[Dict]:
        """Combine and rank results from different methods"""
        if not results:
            return []
        
        # Calculate combined score
        for result in results:
            confidence = result.get('confidence', 0.5)
            similarity = result.get('similarity_score', 0.5)
            quality = result.get('quality_score', confidence)
            
            # Combined ranking score
            combined_score = (
                confidence * 0.4 +
                (1 - similarity) * 0.3 +  # Lower similarity = better paraphrase
                quality * 0.3
            )
            
            result['combined_score'] = combined_score
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Remove duplicates
        unique_results = []
        seen_texts = set()
        
        for result in results:
            text_lower = result['text'].lower()
            if text_lower not in seen_texts:
                unique_results.append(result)
                seen_texts.add(text_lower)
        
        return unique_results[:num_variations]
    
    def paraphrase(self, text: str, method: str = "integrated", num_variations: int = 3, 
                  hybrid_params: Optional[Dict] = None, 
                  t5_params: Optional[Dict] = None) -> List[Dict]:
        """Enhanced paraphrase generation with multiple methods"""
        if not text.strip():
            return []
        
        # Set enhanced default parameters
        if hybrid_params is None:
            hybrid_params = {
                'synonym_rate': 0.4, 
                'use_smart_replacement': True,
                'preserve_keywords': True
            }
        
        if t5_params is None:
            t5_params = {
                'temperature': 0.8,
                'top_p': 0.95,
                'repetition_penalty': 1.3,
                'no_repeat_ngram_size': 3
            }
        
        results = []
        
        if method == "hybrid":
            results = self.paraphrase_hybrid(text, num_variations, **hybrid_params)
            
        elif method == "t5":
            results = self.paraphrase_t5(text, num_variations, t5_params)
            
        elif method == "integrated":
            # Balanced approach
            hybrid_count = max(1, num_variations // 2)
            t5_count = max(1, num_variations - hybrid_count)
            
            if self.enable_hybrid:
                hybrid_results = self.paraphrase_hybrid(text, hybrid_count, **hybrid_params)
                results.extend(hybrid_results)
            
            if self.enable_t5:
                t5_results = self.paraphrase_t5(text, t5_count, t5_params)
                results.extend(t5_results)
                
            # Combine and rank
            results = self.combine_and_rank_results(results, num_variations)
            
        elif method == "best":
            # Generate from all methods and pick the best
            all_results = []
            
            if self.enable_hybrid:
                hybrid_results = self.paraphrase_hybrid(text, num_variations, **hybrid_params)
                all_results.extend(hybrid_results)
            
            if self.enable_t5:
                t5_results = self.paraphrase_t5(text, num_variations, t5_params)
                all_results.extend(t5_results)
            
            # Combine and rank all results
            results = self.combine_and_rank_results(all_results, num_variations)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add final ranking
        for i, result in enumerate(results):
            result['rank'] = i + 1
            result['integrated_method'] = method
        
        return results

# ===================== EXAMPLE USAGE =====================
def main():
    """Example usage of the enhanced paraphraser"""
    # Initialize the integrated paraphraser
    paraphraser = IntegratedParaphraser(
        synonym_dict_path='sinonim.json',
        t5_model_name="LazarusNLP/IndoNanoT5-base",
        enable_hybrid=True,
        enable_t5=True
    )
    
    # Test texts
    test_texts = [
        "Teknologi kecerdasan buatan sangat penting untuk kemajuan masa depan.",
        "Pendidikan adalah kunci utama dalam mengembangkan potensi manusia.",
        "Pemerintah memberikan perhatian khusus terhadap masalah kemiskinan di daerah terpencil."
    ]
    
    print("=" * 80)
    print("ENHANCED INDONESIAN PARAPHRASER DEMO")
    print("=" * 80)
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\n{i}. Original: {test_text}")
        print("-" * 60)
        
        # Test different methods
        methods = ['hybrid', 't5', 'integrated', 'best']
        
        for method in methods:
            if method == 't5' and not paraphraser.enable_t5:
                continue
            if method == 'hybrid' and not paraphraser.enable_hybrid:
                continue
                
            print(f"\n{method.upper()} Method:")
            try:
                results = paraphraser.paraphrase(test_text, method=method, num_variations=2)
                
                for j, result in enumerate(results, 1):
                    print(f"  {j}. {result['text']}")
                    print(f"     Similarity: {result['similarity_score']:.3f} | "
                          f"Confidence: {result['confidence']:.3f} | "
                          f"Method: {result['method']}")
                    
                    if 'replacement_details' in result:
                        replacements = len(result['replacement_details'])
                        print(f"     Replacements: {replacements}")
                        
            except Exception as e:
                print(f"  Error: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
