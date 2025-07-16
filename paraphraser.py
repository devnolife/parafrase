#!/usr/bin/env python3
"""
Sistem Parafrase Bahasa Indonesia - Hybrid + IndoT5 + Integrated
Updated with improved synonym handling
"""

import json
import os
import random
import re
import torch
from typing import List, Dict, Optional, Union
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")

# ===================== HYBRID PARAPHRASER =====================
class HybridParaphraser:
    def __init__(self, synonym_dict_path: str):
        self.synonym_dict = self.load_synonym_dict(synonym_dict_path)
        self.transformation_history = []
        
    def load_synonym_dict(self, path: str) -> Dict:
        """Load synonym dictionary with support for both old and new formats"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check if it's the new format (with 'sinonim' key)
            if data and isinstance(list(data.values())[0], dict) and 'sinonim' in list(data.values())[0]:
                # New format: {"word": {"tag": "v", "sinonim": [...]}}
                return data
            else:
                # Old format: {"word": {"sinonim": [...]}}
                # Convert to new format for consistency
                converted = {}
                for word, value in data.items():
                    if isinstance(value, dict) and 'sinonim' in value:
                        converted[word] = value
                    elif isinstance(value, list):
                        converted[word] = {"sinonim": value, "tag": "unknown"}
                return converted
        except Exception as e:
            print(f"Error loading synonym dictionary: {e}")
            return {}
    
    def preserve_word_format(self, original: str, replacement: str) -> str:
        """Preserve capitalization and punctuation from original word"""
        punctuation = ''
        clean_original = original
        
        # Extract punctuation
        if original and original[-1] in '.,!?;:"':
            punctuation = original[-1]
            clean_original = original[:-1]
        
        # Apply capitalization
        if clean_original.isupper():
            replacement = replacement.upper()
        elif clean_original.istitle():
            replacement = replacement.capitalize()
        elif clean_original.islower():
            replacement = replacement.lower()
        
        return replacement + punctuation
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word from the dictionary"""
        clean_word = word.lower().strip('.,!?;:"')
        if clean_word in self.synonym_dict:
            return self.synonym_dict[clean_word].get('sinonim', [])
        return []
    
    def synonym_replacement(self, text: str, replacement_rate: float = 0.3) -> tuple:
        """Replace words with synonyms based on replacement rate"""
        words = text.split()
        result = []
        replaced_positions = []
        replacement_details = []
        
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?;:"')
            synonyms = self.get_synonyms(clean_word)
            
            if synonyms and random.random() < replacement_rate:
                # Choose a random synonym
                chosen_synonym = random.choice(synonyms)
                formatted_synonym = self.preserve_word_format(word, chosen_synonym)
                result.append(formatted_synonym)
                replaced_positions.append(i)
                replacement_details.append({
                    'original': word,
                    'replacement': formatted_synonym,
                    'position': i,
                    'available_synonyms': synonyms
                })
            else:
                result.append(word)
        
        return ' '.join(result), replaced_positions, replacement_details
    
    def smart_synonym_replacement(self, text: str, replacement_rate: float = 0.3, preserve_keywords: bool = True) -> tuple:
        """Enhanced synonym replacement with keyword preservation"""
        words = text.split()
        result = []
        replaced_positions = []
        replacement_details = []
        
        # Keywords to preserve (can be expanded)
        important_keywords = {'tidak', 'bukan', 'jangan', 'harus', 'wajib', 'penting', 'utama'}
        
        for i, word in enumerate(words):
            clean_word = word.lower().strip('.,!?;:"')
            
            # Skip important keywords if preserve_keywords is True
            if preserve_keywords and clean_word in important_keywords:
                result.append(word)
                continue
            
            synonyms = self.get_synonyms(clean_word)
            
            if synonyms and random.random() < replacement_rate:
                # Filter synonyms to avoid inappropriate replacements
                filtered_synonyms = [s for s in synonyms if len(s.split()) <= 2]  # Avoid very long phrases
                
                if filtered_synonyms:
                    chosen_synonym = random.choice(filtered_synonyms)
                    formatted_synonym = self.preserve_word_format(word, chosen_synonym)
                    result.append(formatted_synonym)
                    replaced_positions.append(i)
                    replacement_details.append({
                        'original': word,
                        'replacement': formatted_synonym,
                        'position': i,
                        'available_synonyms': synonyms,
                        'filtered_synonyms': filtered_synonyms
                    })
                else:
                    result.append(word)
            else:
                result.append(word)
        
        return ' '.join(result), replaced_positions, replacement_details
    
    def active_to_passive(self, sentence: str) -> str:
        """Transform active voice to passive voice"""
        patterns = [
            {
                'pattern': r'(\w+)\s+(me\w+)\s+(\w+)', 
                'transform': lambda m: f"{m.group(3)} di{m.group(2)[2:]} oleh {m.group(1)}"
            },
            {
                'pattern': r'(\w+)\s+(akan|telah)\s+(\w+)\s+(\w+)', 
                'transform': lambda m: f"{m.group(4)} {m.group(2)} di{m.group(3)} oleh {m.group(1)}"
            }
        ]
        
        for p in patterns:
            match = re.search(p['pattern'], sentence)
            if match:
                return p['transform'](match)
        return sentence
    
    def reorder_clauses(self, text: str) -> str:
        """Reorder clauses in compound sentences"""
        if ',' in text:
            parts = text.split(',', 1)
            if len(parts) == 2:
                # Check if first part contains subordinating conjunctions
                subordinating_words = ['karena', 'ketika', 'jika', 'meskipun', 'walaupun', 'setelah', 'sebelum']
                if any(word in parts[0].lower() for word in subordinating_words):
                    return f"{parts[1].strip()}, {parts[0].strip()}"
        return text
    
    def change_conjunctions(self, text: str) -> str:
        """Replace conjunctions with their alternatives"""
        conjunctions = {
            'dan': ['serta', 'juga', 'beserta', 'plus'],
            'atau': ['ataupun', 'maupun'],
            'tetapi': ['namun', 'akan tetapi', 'tapi', 'namun demikian'],
            'karena': ['sebab', 'oleh karena', 'dikarenakan', 'akibat'],
            'jika': ['apabila', 'bila', 'seandainya', 'kalau'],
            'meskipun': ['walaupun', 'sekalipun', 'kendati', 'biarpun'],
            'sehingga': ['hingga', 'sampai', 'maka'],
            'kemudian': ['lalu', 'selanjutnya', 'berikutnya']
        }
        
        result = text
        for original, alternatives in conjunctions.items():
            if original in result.lower():
                replacement = random.choice(alternatives)
                result = re.sub(rf'\b{original}\b', replacement, result, flags=re.IGNORECASE)
                break
        return result
    
    def add_or_remove_modifiers(self, text: str) -> str:
        """Add or remove intensity modifiers"""
        add_modifiers = {
            'penting': 'sangat penting',
            'besar': 'cukup besar',
            'baik': 'lebih baik',
            'jelas': 'sudah jelas',
            'tinggi': 'cukup tinggi',
            'rendah': 'agak rendah'
        }
        
        remove_modifiers = {
            'sangat ': '',
            'cukup ': '',
            'lebih ': '',
            'sudah ': '',
            'agak ': '',
            'lumayan ': ''
        }
        
        if random.random() < 0.5:
            # Add modifiers
            for word, replacement in add_modifiers.items():
                if word in text and replacement not in text:
                    return text.replace(word, replacement)
        else:
            # Remove modifiers
            for modifier, replacement in remove_modifiers.items():
                if modifier in text:
                    return text.replace(modifier, replacement)
        
        return text
    
    def syntactic_transformation(self, text: str) -> str:
        """Apply syntactic transformations"""
        transformations = [
            self.active_to_passive,
            self.reorder_clauses,
            self.change_conjunctions,
            self.add_or_remove_modifiers
        ]
        
        num_transformations = random.randint(1, 2)
        selected_transforms = random.sample(transformations, num_transformations)
        result = text
        
        for transform in selected_transforms:
            new_result = transform(result)
            if new_result != result:
                self.transformation_history.append(transform.__name__)
                result = new_result
        
        return result
    
    def post_process(self, text: str) -> str:
        """Clean up and format the text"""
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?;,])', r'\1', text)
        text = re.sub(r'([.!?;,])\s*', r'\1 ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Capitalize sentences
        sentences = text.split('. ')
        sentences = [s.capitalize() for s in sentences]
        text = '. '.join(sentences)
        
        return text.strip()
    
    def apply_transformations(self, text: str, variation_index: int) -> str:
        """Apply specific transformation sets based on variation index"""
        transformation_sets = [
            [self.active_to_passive, self.change_conjunctions],
            [self.reorder_clauses, self.add_or_remove_modifiers],
            [self.change_conjunctions, self.reorder_clauses, self.active_to_passive]
        ]
        
        transforms = transformation_sets[variation_index % len(transformation_sets)]
        result = text
        
        for transform in transforms:
            new_result = transform(result)
            if new_result != result:
                self.transformation_history.append(transform.__name__)
                result = new_result
        
        return result
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        stop_words = {
            'yang', 'dan', 'atau', 'tetapi', 'karena', 'jika', 'untuk', 'dalam',
            'pada', 'dengan', 'dari', 'ke', 'oleh', 'ini', 'itu', 'adalah',
            'akan', 'telah', 'sudah', 'sedang', 'masih', 'juga', 'hanya',
            'dapat', 'bisa', 'akan', 'harus', 'perlu', 'saja', 'pun'
        }
        
        words = text.lower().split()
        keywords = [word.strip('.,!?;:"') for word in words 
                   if word.strip('.,!?;:"') not in stop_words and len(word.strip('.,!?;:"')) > 2]
        return keywords
    
    def is_valid_paraphrase(self, original: str, paraphrase: str) -> bool:
        """Check if paraphrase is valid"""
        if original.lower() == paraphrase.lower():
            return False
        
        # Check length ratio
        len_ratio = len(paraphrase) / len(original) if len(original) else 0
        if len_ratio < 0.4 or len_ratio > 2.0:
            return False
        
        # Check keyword overlap
        original_keywords = set(self.extract_keywords(original))
        paraphrase_keywords = set(self.extract_keywords(paraphrase))
        
        if not original_keywords:
            return True
        
        overlap = len(original_keywords & paraphrase_keywords) / len(original_keywords)
        if overlap < 0.2:  # Reduced threshold for more variety
            return False
        
        return True
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0
    
    def paraphrase(self, text: str, num_variations: int = 3, synonym_rate: float = 0.3, 
                  use_smart_replacement: bool = True) -> List[Dict]:
        """Generate paraphrases with enhanced synonym replacement"""
        paraphrases = []
        
        for i in range(num_variations):
            self.transformation_history = []
            processed_text = ' '.join(text.split())
            
            # Adjust synonym replacement rate for each variation
            current_rate = min(synonym_rate + (i * 0.1), 0.8)
            
            # Use smart or regular synonym replacement
            if use_smart_replacement:
                text_with_synonyms, replaced_positions, replacement_details = self.smart_synonym_replacement(
                    processed_text, current_rate
                )
            else:
                text_with_synonyms, replaced_positions, replacement_details = self.synonym_replacement(
                    processed_text, current_rate
                )
            
            # Apply syntactic transformations
            transformed_text = self.apply_transformations(text_with_synonyms, variation_index=i)
            
            # Post-process the text
            final_text = self.post_process(transformed_text)
            
            # Validate paraphrase
            if self.is_valid_paraphrase(text, final_text):
                paraphrases.append({
                    'text': final_text,
                    'transformations': self.transformation_history.copy(),
                    'similarity_score': self.calculate_similarity(text, final_text),
                    'replaced_positions': replaced_positions,
                    'replacement_details': replacement_details,
                    'synonym_rate_used': current_rate
                })
        
        return paraphrases

# ===================== INDOT5 PARAPHRASER =====================
class IndoT5Paraphraser:
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
        """Load the T5 model and tokenizer"""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                return True
            except Exception:
                return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for T5 model"""
        text = text.strip()
        text = ' '.join(text.split())
        
        if "paraphraser" in self.model_name.lower():
            return f"paraphrase: {text} </s>"
        elif "summarization" in self.model_name.lower():
            return f"summarize: {text} </s>"
        else:
            return f"paraphrase: {text}"
    
    def postprocess_text(self, text: str) -> str:
        """Postprocess T5 model output"""
        text = text.strip()
        
        # Remove common prefixes
        prefixes = ["paraphrased output:", "paraphrase:", "summarize:", "summary:"]
        for prefix in prefixes:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Capitalize first letter
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        return text
    
    def generate_paraphrases(self, text: str, num_variations: int = 3, 
                           generation_params: Optional[Dict] = None) -> List[str]:
        """Generate paraphrases using T5 model"""
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        default_params = {
            "max_length": 128,
            "num_beams": 5,
            "num_return_sequences": num_variations,
            "early_stopping": True,
            "do_sample": True,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "no_repeat_ngram_size": 2,
            "repetition_penalty": 1.2
        }
        
        if generation_params:
            default_params.update(generation_params)
        
        try:
            processed_text = self.preprocess_text(text)
            inputs = self.tokenizer.encode(
                processed_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(inputs, **default_params)
            
            paraphrases = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                processed = self.postprocess_text(decoded)
                if processed and processed.lower() != text.lower():
                    paraphrases.append(processed)
            
            # Remove duplicates while preserving order
            unique_paraphrases = []
            for para in paraphrases:
                if para not in unique_paraphrases:
                    unique_paraphrases.append(para)
            
            return unique_paraphrases[:num_variations]
            
        except Exception:
            return []
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0
    
    def paraphrase(self, text: str, num_variations: int = 3, 
                  generation_params: Optional[Dict] = None) -> List[Dict]:
        """Generate paraphrases and return structured results"""
        paraphrases = self.generate_paraphrases(text, num_variations, generation_params)
        results = []
        
        for i, paraphrase in enumerate(paraphrases):
            result = {
                'text': paraphrase,
                'method': 'IndoT5',
                'model': self.model_name,
                'similarity_score': self.calculate_similarity(text, paraphrase),
                'confidence': 1.0 - (i * 0.1),
                'generation_rank': i + 1
            }
            results.append(result)
        
        return results

# ===================== INTEGRATED PARAPHRASER =====================
class IntegratedParaphraser:
    def __init__(self, synonym_dict_path: str = 'sinonim.json', 
                 t5_model_name: str = "LazarusNLP/IndoNanoT5-base", 
                 enable_hybrid: bool = True, enable_t5: bool = True):
        self.enable_hybrid = enable_hybrid
        self.enable_t5 = enable_t5
        
        # Initialize hybrid paraphraser
        self.hybrid_paraphraser = None
        if enable_hybrid and os.path.exists(synonym_dict_path):
            try:
                self.hybrid_paraphraser = HybridParaphraser(synonym_dict_path)
                print(f"Hybrid paraphraser loaded with {len(self.hybrid_paraphraser.synonym_dict)} synonyms")
            except Exception as e:
                print(f"Failed to load hybrid paraphraser: {e}")
                self.enable_hybrid = False
        
        # Initialize T5 paraphraser
        self.t5_paraphraser = None
        if enable_t5:
            try:
                self.t5_paraphraser = IndoT5Paraphraser(t5_model_name)
                print(f"T5 paraphraser initialized with model: {t5_model_name}")
            except Exception as e:
                print(f"Failed to initialize T5 paraphraser: {e}")
                self.enable_t5 = False
        
        self.methods_available = {
            'hybrid': self.enable_hybrid,
            't5': self.enable_t5
        }
    
    def paraphrase_hybrid(self, text: str, num_variations: int = 3, 
                         synonym_rate: float = 0.3, use_smart_replacement: bool = True) -> List[Dict]:
        """Generate paraphrases using hybrid method"""
        if not self.enable_hybrid or not self.hybrid_paraphraser:
            return []
        
        try:
            results = self.hybrid_paraphraser.paraphrase(
                text, num_variations, synonym_rate, use_smart_replacement
            )
            
            # Add method metadata
            for result in results:
                result['method'] = 'Hybrid'
                result['model'] = 'Rule-based + Synonyms'
                if 'confidence' not in result:
                    result['confidence'] = result.get('similarity_score', 0.8)
            
            return results
        except Exception as e:
            print(f"Error in hybrid paraphrasing: {e}")
            return []
    
    def paraphrase_t5(self, text: str, num_variations: int = 3, 
                     generation_params: Optional[Dict] = None) -> List[Dict]:
        """Generate paraphrases using T5 method"""
        if not self.enable_t5 or not self.t5_paraphraser:
            return []
        
        try:
            return self.t5_paraphraser.paraphrase(text, num_variations, generation_params)
        except Exception as e:
            print(f"Error in T5 paraphrasing: {e}")
            return []
    
    def paraphrase(self, text: str, method: str = "integrated", num_variations: int = 3, 
                  hybrid_params: Optional[Dict] = None, 
                  t5_params: Optional[Dict] = None) -> List[Dict]:
        """Generate paraphrases using specified method"""
        if not text.strip():
            return []
        
        # Set default parameters
        if hybrid_params is None:
            hybrid_params = {'synonym_rate': 0.3, 'use_smart_replacement': True}
        if t5_params is None:
            t5_params = {}
        
        results = []
        
        if method == "hybrid":
            results = self.paraphrase_hybrid(text, num_variations, **hybrid_params)
            
        elif method == "t5":
            results = self.paraphrase_t5(text, num_variations, t5_params)
            
        elif method == "integrated":
            # Split variations between methods
            hybrid_count = num_variations // 2
            t5_count = num_variations - hybrid_count
            
            if self.enable_hybrid:
                hybrid_results = self.paraphrase_hybrid(text, hybrid_count, **hybrid_params)
                results.extend(hybrid_results)
            
            if self.enable_t5:
                t5_results = self.paraphrase_t5(text, t5_count, t5_params)
                results.extend(t5_results)
                
        elif method == "best":
            # Generate results from all available methods
            all_results = []
            
            if self.enable_hybrid:
                hybrid_results = self.paraphrase_hybrid(text, num_variations, **hybrid_params)
                all_results.extend(hybrid_results)
            
            if self.enable_t5:
                t5_results = self.paraphrase_t5(text, num_variations, t5_params)
                all_results.extend(t5_results)
            
            # Sort by confidence and similarity score
            all_results.sort(
                key=lambda x: (x.get('confidence', 0), x.get('similarity_score', 0)), 
                reverse=True
            )
            
            # Remove duplicates
            unique_results = []
            seen_texts = set()
            for result in all_results:
                if result['text'].lower() not in seen_texts:
                    unique_results.append(result)
                    seen_texts.add(result['text'].lower())
            
            results = unique_results[:num_variations]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add rank and method metadata
        for i, result in enumerate(results):
            result['rank'] = i + 1
            result['integrated_method'] = method
        
        return results

# ===================== EXAMPLE USAGE =====================
def main():
    """Example usage of the paraphraser"""
    # Initialize the integrated paraphraser
    paraphraser = IntegratedParaphraser(
        synonym_dict_path='sinonim.json',
        t5_model_name="LazarusNLP/IndoNanoT5-base",
        enable_hybrid=True,
        enable_t5=False  # Set to False to avoid downloading model for demo
    )
    
    # Test text
    test_text = "Teknologi kecerdasan buatan sangat penting untuk kemajuan masa depan."
    
    print(f"Original text: {test_text}")
    print("=" * 50)
    
    # Test hybrid method
    if paraphraser.enable_hybrid:
        print("Hybrid Method Results:")
        hybrid_results = paraphraser.paraphrase(test_text, method="hybrid", num_variations=3)
        for i, result in enumerate(hybrid_results, 1):
            print(f"{i}. {result['text']}")
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Transformations: {result.get('transformations', [])}")
            print(f"   Replacements: {len(result.get('replacement_details', []))}")
            print()
    
    # Test integrated method
    print("Integrated Method Results:")
    integrated_results = paraphraser.paraphrase(test_text, method="integrated", num_variations=3)
    for i, result in enumerate(integrated_results, 1):
        print(f"{i}. {result['text']}")
        print(f"   Method: {result['method']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print()

if __name__ == "__main__":
    main()
