#!/usr/bin/env python3
"""
Unit tests untuk Paraphraser (Hybrid, IndoT5, Integrated)
"""

import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
from paraphraser import HybridParaphraser, IndoT5Paraphraser, IntegratedParaphraser

class TestParaphraser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_synonym_file = 'test_sinonim.json'
        test_synonyms = {
            "penting": {"sinonim": ["vital", "esensial", "krusial"]},
            "baik": {"sinonim": ["bagus", "unggul", "hebat"]},
            "belajar": {"sinonim": ["menimba ilmu", "mempelajari", "mengkaji"]},
            "pendidikan": {"sinonim": ["edukasi", "pembelajaran", "pengajaran"]}
        }
        with open(cls.test_synonym_file, 'w', encoding='utf-8') as f:
            json.dump(test_synonyms, f, ensure_ascii=False, indent=2)
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_synonym_file):
            os.remove(cls.test_synonym_file)
    def setUp(self):
        self.test_text = "Pendidikan adalah hal yang sangat penting untuk masa depan."
    def test_hybrid_paraphrase(self):
        paraphraser = HybridParaphraser(self.test_synonym_file)
        results = paraphraser.paraphrase(self.test_text, num_variations=2)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) <= 2)
        for result in results:
            self.assertIn('text', result)
            self.assertIn('similarity_score', result)
            self.assertNotEqual(result['text'].lower(), self.test_text.lower())
    @patch('paraphraser.T5ForConditionalGeneration')
    @patch('paraphraser.T5Tokenizer')
    @patch('paraphraser.torch')
    def test_indot5_paraphrase(self, mock_torch, mock_tokenizer_class, mock_model_class):
        mock_torch.cuda.is_available.return_value = False
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        paraphraser = IndoT5Paraphraser("test-model")
        paraphraser.is_loaded = True
        paraphraser.tokenizer = mock_tokenizer
        paraphraser.model = mock_model
        mock_model.generate.return_value = [MagicMock()]
        mock_tokenizer.decode.return_value = "Edukasi merupakan hal yang amat vital untuk masa depan."
        results = paraphraser.paraphrase(self.test_text, num_variations=1)
        self.assertIsInstance(results, list)
        self.assertEqual(results[0]['method'], 'IndoT5')
    @patch('paraphraser.IndoT5Paraphraser')
    def test_integrated_paraphraser(self, mock_t5_class):
        mock_t5_instance = MagicMock()
        mock_t5_instance.paraphrase.return_value = [
            {'text': 'Edukasi merupakan hal yang amat vital untuk masa depan.', 'method': 'IndoT5', 'model': 'test-model', 'similarity_score': 0.85, 'confidence': 0.9}
        ]
        mock_t5_class.return_value = mock_t5_instance
        paraphraser = IntegratedParaphraser(self.test_synonym_file, enable_hybrid=True, enable_t5=True)
        results = paraphraser.paraphrase(self.test_text, method="integrated", num_variations=2)
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) <= 2)
        methods_found = set(result['method'] for result in results)
        self.assertTrue(len(methods_found) >= 1)
    def test_invalid_method(self):
        paraphraser = IntegratedParaphraser(self.test_synonym_file, enable_hybrid=True, enable_t5=False)
        with self.assertRaises(ValueError):
            paraphraser.paraphrase(self.test_text, method="invalid_method")
    def test_empty_text(self):
        paraphraser = IntegratedParaphraser(self.test_synonym_file, enable_hybrid=True, enable_t5=False)
        results = paraphraser.paraphrase("", method="hybrid")
        self.assertEqual(len(results), 0)
        results = paraphraser.paraphrase("   ", method="hybrid")
        self.assertEqual(len(results), 0)
if __name__ == '__main__':
    unittest.main(verbosity=2)
