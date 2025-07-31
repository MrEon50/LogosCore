#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 LOGOS-ASM — Terminal Świadomości Kodu
Semantyczny System Uczenia Kodu Maszynowego z DRM
"""

import os
import sys
import json
import math
import random
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle


@dataclass
class Rule:
    """Reguła poznawcza w Dynamic Rule Matrix"""
    id: int
    pattern: str
    semantic: str
    weight: float = 0.5
    count: int = 0
    activity: float = 0.0
    time_created: int = 0
    correlation: float = 0.5
    last_used: int = 0
    success_rate: float = 0.5
    
    def calculate_strength(self, current_time: int) -> float:
        """Oblicza siłę reguły według wzoru DRM"""
        if self.count == 0:
            return 0.0
        
        time_factor = 1 + (self.activity / max(current_time, 1))
        strength = (self.weight * 
                   math.log(self.count + 1) * 
                   time_factor * 
                   self.correlation)
        return max(0.0, strength)


class InstructionParser:
    """Parser instrukcji assemblera i kodu maszynowego"""
    
    def __init__(self):
        self.asm_patterns = {
            r'MOV\s+(\w+),\s*(\w+)': 'transfer_data',
            r'ADD\s+(\w+),\s*(\w+)': 'arithmetic_add',
            r'SUB\s+(\w+),\s*(\w+)': 'arithmetic_sub',
            r'MUL\s+(\w+),\s*(\w+)': 'arithmetic_mul',
            r'DIV\s+(\w+),\s*(\w+)': 'arithmetic_div',
            r'CMP\s+(\w+),\s*(\w+)': 'compare_values',
            r'JMP\s+(\w+)': 'unconditional_jump',
            r'JE\s+(\w+)': 'jump_if_equal',
            r'JNE\s+(\w+)': 'jump_if_not_equal',
            r'JL\s+(\w+)': 'jump_if_less',
            r'JG\s+(\w+)': 'jump_if_greater',
            r'PUSH\s+(\w+)': 'stack_push',
            r'POP\s+(\w+)': 'stack_pop',
            r'CALL\s+(\w+)': 'function_call',
            r'RET': 'function_return',
            r'NOP': 'no_operation',
            r'INT\s+(\w+)': 'interrupt_call',
            r'LOAD\s+(\w+),\s*(\w+)': 'memory_load',
            r'STORE\s+(\w+),\s*(\w+)': 'memory_store'
        }
        
        self.semantic_meanings = {
            'transfer_data': 'przeniesienie danych między rejestrami/pamięcią',
            'arithmetic_add': 'operacja dodawania arytmetycznego',
            'arithmetic_sub': 'operacja odejmowania arytmetycznego',
            'arithmetic_mul': 'operacja mnożenia arytmetycznego',
            'arithmetic_div': 'operacja dzielenia arytmetycznego',
            'compare_values': 'porównanie wartości i ustawienie flag',
            'unconditional_jump': 'bezwarunkowy skok do etykiety',
            'jump_if_equal': 'skok warunkowy jeśli równe',
            'jump_if_not_equal': 'skok warunkowy jeśli różne',
            'jump_if_less': 'skok warunkowy jeśli mniejsze',
            'jump_if_greater': 'skok warunkowy jeśli większe',
            'stack_push': 'umieszczenie wartości na stosie',
            'stack_pop': 'pobranie wartości ze stosu',
            'function_call': 'wywołanie funkcji/procedury',
            'function_return': 'powrót z funkcji',
            'no_operation': 'brak operacji - pauza',
            'interrupt_call': 'wywołanie przerwania systemowego',
            'memory_load': 'ładowanie danych z pamięci',
            'memory_store': 'zapisywanie danych do pamięci'
        }

    def parse_instruction(self, instruction: str) -> Tuple[Optional[str], Optional[str]]:
        """Parsuje instrukcję i zwraca typ oraz semantykę"""
        instruction = instruction.strip().upper()
        
        for pattern, instr_type in self.asm_patterns.items():
            if re.match(pattern, instruction):
                semantic = self.semantic_meanings.get(instr_type, 'nieznana operacja')
                return instr_type, semantic
        
        return None, None

    def binary_to_asm(self, binary_str: str) -> str:
        """Symuluje konwersję kodu binarnego na assembler"""
        if not binary_str or len(binary_str) % 8 != 0:
            return "INVALID"
        
        # Uproszczona symulacja - mapowanie wzorców binarnych
        binary_patterns = {
            '10110000': 'MOV AL, imm8',
            '10001000': 'MOV r/m8, r8',
            '00000100': 'ADD AL, imm8',
            '00101100': 'SUB AL, imm8',
            '00111100': 'CMP AL, imm8',
            '11101001': 'JMP rel32',
            '01010000': 'PUSH EAX',
            '01011000': 'POP EAX',
            '11101000': 'CALL rel32',
            '11000011': 'RET',
            '10010000': 'NOP'
        }
        
        chunks = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]
        asm_instructions = []
        
        for chunk in chunks:
            asm = binary_patterns.get(chunk, f'DB {chunk}')
            asm_instructions.append(asm)
        
        return '\n'.join(asm_instructions)


class DeepSemanticNeuralNetwork:
    """Głęboka sieć neuronowa dla analizy semantycznej kodu"""

    def __init__(self, input_size=128, hidden_sizes=[256, 128, 64], output_size=32):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = 0.001
        self.momentum = 0.9

        # Inicjalizacja wag i biasów
        self.weights = []
        self.biases = []
        self.velocities_w = []
        self.velocities_b = []

        # Warstwy: input -> hidden1 -> hidden2 -> hidden3 -> output
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Xavier/Glorot initialization
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))

            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

            # Momentum velocities
            self.velocities_w.append(np.zeros_like(w))
            self.velocities_b.append(np.zeros_like(b))

        # Semantic embeddings cache
        self.semantic_cache = {}
        self.instruction_embeddings = {}

        # Training history
        self.loss_history = []
        self.accuracy_history = []

    def sigmoid(self, x):
        """Funkcja aktywacji sigmoid z zabezpieczeniem przed overflow"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def relu(self, x):
        """Funkcja aktywacji ReLU"""
        return np.maximum(0, x)

    def leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation"""
        return np.where(x > 0, x, alpha * x)

    def softmax(self, x):
        """Softmax activation dla warstwy wyjściowej"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        """Forward pass przez sieć"""
        activations = [x]

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b

            if i < len(self.weights) - 1:  # Hidden layers
                if i == 0:
                    a = self.relu(z)  # Pierwsza warstwa: ReLU
                else:
                    a = self.leaky_relu(z)  # Pozostałe: Leaky ReLU
            else:  # Output layer
                a = self.sigmoid(z)  # Wyjście: Sigmoid

            activations.append(a)

        return activations

    def encode_instruction(self, instruction: str, instr_type: str = None) -> np.ndarray:
        """Koduje instrukcję do wektora cech"""
        # Cache dla wydajności
        cache_key = f"{instruction}_{instr_type}"
        if cache_key in self.instruction_embeddings:
            return self.instruction_embeddings[cache_key]

        # Podstawowe cechy instrukcji
        features = np.zeros(self.input_size)

        # 1. Cechy leksykalne (0-31)
        instruction_upper = instruction.upper()

        # Podstawowe instrukcje
        basic_instructions = ['MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'CMP', 'JMP', 'CALL',
                             'RET', 'PUSH', 'POP', 'NOP', 'INT', 'LOAD', 'STORE']
        for i, instr in enumerate(basic_instructions):
            if instr in instruction_upper:
                features[i] = 1.0

        # 2. Cechy syntaktyczne (32-63)
        # Rejestry
        registers = ['A', 'B', 'C', 'D', 'EAX', 'EBX', 'ECX', 'EDX', 'ESP', 'EBP']
        for i, reg in enumerate(registers):
            if reg in instruction_upper:
                features[32 + i] = 1.0

        # Liczby i adresy
        if re.search(r'\d+', instruction):
            features[42] = 1.0  # Zawiera liczby
        if '[' in instruction and ']' in instruction:
            features[43] = 1.0  # Adresowanie pośrednie
        if ',' in instruction:
            features[44] = 1.0  # Wiele operandów

        # 3. Cechy semantyczne (64-95)
        if instr_type:
            semantic_types = ['transfer_data', 'arithmetic_add', 'arithmetic_sub',
                             'arithmetic_mul', 'arithmetic_div', 'compare_values',
                             'unconditional_jump', 'jump_if_equal', 'jump_if_not_equal',
                             'stack_push', 'stack_pop', 'function_call', 'function_return']

            for i, sem_type in enumerate(semantic_types):
                if sem_type in instr_type:
                    features[64 + i] = 1.0

        # 4. Cechy kontekstowe (96-127)
        # Długość instrukcji
        features[96] = min(len(instruction) / 50.0, 1.0)

        # Złożoność (liczba tokenów)
        tokens = instruction.split()
        features[97] = min(len(tokens) / 10.0, 1.0)

        # Hash instrukcji dla unikalności
        instr_hash = hash(instruction) % 30
        features[98 + instr_hash % 30] = 1.0

        # Cache result
        self.instruction_embeddings[cache_key] = features
        return features

    def train_batch(self, instructions, targets, epochs=1):
        """Trenuje sieć na batch'u danych"""
        X = np.array([self.encode_instruction(instr[0], instr[1]) for instr in instructions])
        y = np.array(targets)

        for epoch in range(epochs):
            # Forward pass
            activations = self.forward(X)
            predictions = activations[-1]

            # Compute loss (Mean Squared Error)
            loss = np.mean((predictions - y) ** 2)
            self.loss_history.append(loss)

            # Compute accuracy
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1)
            accuracy = np.mean(pred_classes == true_classes)
            self.accuracy_history.append(accuracy)

            # Backward pass
            self._backward(activations, y)

    def _backward(self, activations, y):
        """Backward pass - backpropagation"""
        m = y.shape[0]  # batch size

        # Compute gradients
        dW = []
        db = []

        # Output layer error
        dA = activations[-1] - y  # derivative of MSE

        for i in reversed(range(len(self.weights))):
            # Gradient of weights and biases
            dW_i = np.dot(activations[i].T, dA) / m
            db_i = np.sum(dA, axis=0, keepdims=True) / m

            dW.insert(0, dW_i)
            db.insert(0, db_i)

            if i > 0:  # Not input layer
                # Gradient of activation
                dZ = np.dot(dA, self.weights[i].T)

                # Apply derivative of activation function
                if i == 1:  # First hidden layer (ReLU)
                    dA = dZ * (activations[i] > 0)
                else:  # Other hidden layers (Leaky ReLU)
                    dA = dZ * np.where(activations[i] > 0, 1, 0.01)

        # Update weights with momentum
        for i in range(len(self.weights)):
            # Momentum update
            self.velocities_w[i] = (self.momentum * self.velocities_w[i] -
                                   self.learning_rate * dW[i])
            self.velocities_b[i] = (self.momentum * self.velocities_b[i] -
                                   self.learning_rate * db[i])

            # Apply updates
            self.weights[i] += self.velocities_w[i]
            self.biases[i] += self.velocities_b[i]

    def predict(self, instruction: str, instr_type: str = None) -> np.ndarray:
        """Predykcja dla pojedynczej instrukcji"""
        x = self.encode_instruction(instruction, instr_type).reshape(1, -1)
        activations = self.forward(x)
        return activations[-1][0]

    def get_semantic_similarity(self, instr1: str, instr2: str,
                               type1: str = None, type2: str = None) -> float:
        """Oblicza podobieństwo semantyczne między instrukcjami"""
        emb1 = self.encode_instruction(instr1, type1)
        emb2 = self.encode_instruction(instr2, type2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def analyze_instruction_complexity(self, instruction: str, instr_type: str = None) -> Dict:
        """Analizuje złożoność instrukcji"""
        embedding = self.encode_instruction(instruction, instr_type)
        prediction = self.predict(instruction, instr_type)

        # Analiza różnych aspektów
        lexical_complexity = np.sum(embedding[:32])  # Cechy leksykalne
        syntactic_complexity = np.sum(embedding[32:64])  # Cechy syntaktyczne
        semantic_complexity = np.sum(embedding[64:96])  # Cechy semantyczne
        contextual_complexity = np.sum(embedding[96:])  # Cechy kontekstowe

        # Entropia predykcji (miara niepewności)
        entropy = -np.sum(prediction * np.log(prediction + 1e-10))

        return {
            'lexical_complexity': float(lexical_complexity),
            'syntactic_complexity': float(syntactic_complexity),
            'semantic_complexity': float(semantic_complexity),
            'contextual_complexity': float(contextual_complexity),
            'prediction_entropy': float(entropy),
            'total_complexity': float(np.sum(embedding)),
            'confidence': float(np.max(prediction))
        }

    def get_network_state(self) -> Dict:
        """Zwraca stan sieci neuronowej"""
        return {
            'total_parameters': sum(w.size for w in self.weights) + sum(b.size for b in self.biases),
            'layer_sizes': [self.input_size] + self.hidden_sizes + [self.output_size],
            'training_samples': len(self.loss_history),
            'current_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'current_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'cache_size': len(self.instruction_embeddings)
        }


class DynamicRuleMatrix:
    """System zarządzania regułami poznawczymi"""

    def __init__(self):
        self.rules: Dict[int, Rule] = {}
        self.rule_counter = 0
        self.current_time = 0
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
    def add_rule(self, pattern: str, semantic: str, weight: float = 0.5) -> int:
        """Dodaje nową regułę do macierzy"""
        self.rule_counter += 1
        rule = Rule(
            id=self.rule_counter,
            pattern=pattern,
            semantic=semantic,
            weight=weight,
            time_created=self.current_time
        )
        self.rules[self.rule_counter] = rule
        return self.rule_counter
    
    def update_rule(self, rule_id: int, success: bool = True, activity_boost: float = 1.0):
        """Aktualizuje regułę na podstawie użycia"""
        if rule_id not in self.rules:
            return
        
        rule = self.rules[rule_id]
        rule.count += 1
        rule.activity += activity_boost
        rule.last_used = self.current_time
        
        if success:
            rule.success_rate = (rule.success_rate * 0.9) + (1.0 * 0.1)
            rule.correlation = min(1.0, rule.correlation + self.learning_rate)
            rule.weight = min(1.0, rule.weight + self.learning_rate * 0.5)
        else:
            rule.success_rate = (rule.success_rate * 0.9) + (0.0 * 0.1)
            rule.correlation = max(0.0, rule.correlation - self.learning_rate)
            rule.weight = max(0.0, rule.weight - self.learning_rate * 0.3)
    
    def decay_rules(self):
        """Stosuje zanik dla nieużywanych reguł"""
        for rule in self.rules.values():
            time_since_use = self.current_time - rule.last_used
            if time_since_use > 10:
                rule.weight *= self.decay_factor
                rule.activity *= self.decay_factor
    
    def get_strongest_rules(self, limit: int = 10) -> List[Rule]:
        """Zwraca najsilniejsze reguły"""
        rules_with_strength = [
            (rule, rule.calculate_strength(self.current_time))
            for rule in self.rules.values()
        ]
        rules_with_strength.sort(key=lambda x: x[1], reverse=True)
        return [rule for rule, _ in rules_with_strength[:limit]]
    
    def remove_weak_rules(self, threshold: float = 0.1):
        """Usuwa reguły o niskiej sile"""
        to_remove = []
        for rule_id, rule in self.rules.items():
            if rule.calculate_strength(self.current_time) < threshold:
                to_remove.append(rule_id)
        
        for rule_id in to_remove:
            del self.rules[rule_id]
        
        return len(to_remove)
    
    def tick(self):
        """Aktualizuje czas systemowy"""
        self.current_time += 1
        if self.current_time % 50 == 0:
            self.decay_rules()


class ProcessorSimulator:
    """Symulator procesora dla kodu maszynowego"""
    
    def __init__(self):
        self.registers = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'EAX': 0, 'EBX': 0}
        self.memory = [0] * 1024
        self.stack = []
        self.flags = {'zero': False, 'carry': False, 'negative': False}
        self.program_counter = 0
        self.execution_log = []
    
    def reset(self):
        """Resetuje stan procesora"""
        self.registers = {k: 0 for k in self.registers}
        self.memory = [0] * 1024
        self.stack = []
        self.flags = {'zero': False, 'carry': False, 'negative': False}
        self.program_counter = 0
        self.execution_log = []
    
    def execute_instruction(self, instruction: str) -> bool:
        """Wykonuje pojedynczą instrukcję"""
        instruction = instruction.strip().upper()
        self.execution_log.append(f"PC:{self.program_counter:04d} | {instruction}")
        
        try:
            # MOV
            if match := re.match(r'MOV\s+(\w+),\s*(\w+)', instruction):
                dest, src = match.groups()
                if src.isdigit():
                    self.registers[dest] = int(src)
                else:
                    self.registers[dest] = self.registers.get(src, 0)
            
            # ADD
            elif match := re.match(r'ADD\s+(\w+),\s*(\w+)', instruction):
                dest, src = match.groups()
                val = int(src) if src.isdigit() else self.registers.get(src, 0)
                result = self.registers[dest] + val
                self.registers[dest] = result & 0xFFFFFFFF
                self.flags['zero'] = (result == 0)
                self.flags['carry'] = (result > 0xFFFFFFFF)
            
            # SUB
            elif match := re.match(r'SUB\s+(\w+),\s*(\w+)', instruction):
                dest, src = match.groups()
                val = int(src) if src.isdigit() else self.registers.get(src, 0)
                result = self.registers[dest] - val
                self.registers[dest] = result & 0xFFFFFFFF
                self.flags['zero'] = (result == 0)
                self.flags['negative'] = (result < 0)
            
            # CMP
            elif match := re.match(r'CMP\s+(\w+),\s*(\w+)', instruction):
                reg1, reg2 = match.groups()
                val1 = self.registers.get(reg1, 0)
                val2 = int(reg2) if reg2.isdigit() else self.registers.get(reg2, 0)
                result = val1 - val2
                self.flags['zero'] = (result == 0)
                self.flags['negative'] = (result < 0)
            
            # PUSH
            elif match := re.match(r'PUSH\s+(\w+)', instruction):
                reg = match.group(1)
                self.stack.append(self.registers.get(reg, 0))
            
            # POP
            elif match := re.match(r'POP\s+(\w+)', instruction):
                reg = match.group(1)
                if self.stack:
                    self.registers[reg] = self.stack.pop()
            
            # NOP
            elif instruction == 'NOP':
                pass  # No operation
            
            else:
                self.execution_log.append(f"  BŁĄD: Nieznana instrukcja")
                return False
            
            self.program_counter += 1
            return True
            
        except Exception as e:
            self.execution_log.append(f"  BŁĄD WYKONANIA: {e}")
            return False
    
    def get_state(self) -> Dict:
        """Zwraca aktualny stan procesora"""
        return {
            'registers': self.registers.copy(),
            'flags': self.flags.copy(),
            'stack_size': len(self.stack),
            'stack_top': self.stack[-1] if self.stack else None,
            'program_counter': self.program_counter
        }


class LogosASM:
    """Główny system LOGOS-ASM z Deep Semantic Neural Network"""

    def __init__(self):
        self.drm = DynamicRuleMatrix()
        self.parser = InstructionParser()
        self.processor = ProcessorSimulator()
        self.dsnn = DeepSemanticNeuralNetwork()  # 🧠 Nowa sieć neuronowa

        self.training_data = []
        self.neural_training_data = []  # Dane dla sieci neuronowej
        self.session_stats = {
            'instructions_processed': 0,
            'rules_created': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'neural_predictions': 0,
            'hybrid_predictions': 0
        }
        self.initialize_basic_rules()
        self.training_history = []
        self.metrics_log = []
        self.learning_curves = {
            'accuracy': [],
            'rule_strength': [],
            'prediction_confidence': [],
            'neural_loss': [],
            'neural_accuracy': [],
            'hybrid_performance': []
        }

        # Hybrid prediction system
        self.hybrid_mode = True
        self.neural_weight = 0.3  # Waga sieci neuronowej w predykcji hybrydowej
        self.drm_weight = 0.7     # Waga DRM w predykcji hybrydowej
    
    def initialize_basic_rules(self):
        """Inicjalizuje podstawowe reguły"""
        basic_rules = [
            ('MOV', 'transfer danych między lokalizacjami', 0.8),
            ('ADD', 'operacja arytmetyczna dodawania', 0.7),
            ('SUB', 'operacja arytmetyczna odejmowania', 0.7),
            ('CMP', 'porównanie wartości z ustawieniem flag', 0.75),
            ('JMP', 'bezwarunkowy skok w kodzie', 0.6),
            ('PUSH', 'umieszczenie na stosie', 0.65),
            ('POP', 'pobranie ze stosu', 0.65),
            ('CALL', 'wywołanie procedury', 0.8),
            ('RET', 'powrót z procedury', 0.8)
        ]
        
        for pattern, semantic, weight in basic_rules:
            self.drm.add_rule(pattern, semantic, weight)
    
    def train_on_code(self, code: str, code_type: str = 'asm'):
        """Trenuje system na podanym kodzie - Z DEBUGIEM"""
        lines = code.strip().split('\n')
        
        print(f"\n🔄 ROZPOCZYNAM TRENING ({code_type}):")
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue
            
            print(f"\n📝 Linia {i}: {line}")
            
            if code_type == 'binary':
                asm_line = self.parser.binary_to_asm(line)
                if asm_line != "INVALID":
                    self._process_instruction(asm_line)
            else:
                self._process_instruction(line)
            
            self.session_stats['instructions_processed'] += 1
        
        self.drm.tick()
        
        print(f"\n✅ TRENING ZAKOŃCZONY:")
        print(f"📊 Utworzono {len(self.drm.rules)} reguł")
        self.debug_rules()  # Pokaż wszystkie reguły
    
    def _process_instruction(self, instruction: str):
        """Przetwarza pojedynczą instrukcję - HYBRYDOWA WERSJA z DSNN"""
        instr_type, semantic = self.parser.parse_instruction(instruction)

        # 🧠 NEURAL NETWORK PREDICTION
        neural_prediction = None
        neural_confidence = 0.0

        if self.hybrid_mode:
            try:
                neural_output = self.dsnn.predict(instruction, instr_type)
                neural_confidence = float(np.max(neural_output))
                neural_prediction = self._interpret_neural_output(neural_output)
                self.session_stats['neural_predictions'] += 1
            except Exception as e:
                print(f"  ⚠️ Neural prediction failed: {e}")

        # 🧮 DRM PROCESSING (original logic)
        if instr_type and semantic:
            rule_found = False
            drm_confidence = 0.0

            for rule in self.drm.rules.values():
                if (instr_type.lower() in rule.pattern.lower() or
                    rule.pattern.lower() in instr_type.lower()):
                    self.drm.update_rule(rule.id, True, 1.0)
                    rule_found = True
                    drm_confidence = rule.calculate_strength(self.drm.current_time)
                    print(f"  🔄 Zaktualizowano regułę #{rule.id}: {rule.pattern}")
                    break

            if not rule_found:
                rule_id = self.drm.add_rule(instr_type, semantic, 0.6)
                self.drm.update_rule(rule_id, True, 1.0)
                self.session_stats['rules_created'] += 1
                drm_confidence = 0.6
                print(f"  ✨ Utworzono nową regułę #{rule_id}: {instr_type}")

            # 🤝 HYBRID PREDICTION FUSION
            if self.hybrid_mode and neural_prediction:
                hybrid_confidence = (self.drm_weight * drm_confidence +
                                   self.neural_weight * neural_confidence)

                print(f"  🤖 Neural: {neural_prediction} (conf: {neural_confidence:.2f})")
                print(f"  🧮 DRM: {instr_type} (conf: {drm_confidence:.2f})")
                print(f"  🤝 Hybrid confidence: {hybrid_confidence:.2f}")

                self.session_stats['hybrid_predictions'] += 1

                # Adaptacyjne dostrajanie wag
                if neural_confidence > drm_confidence:
                    self.neural_weight = min(0.5, self.neural_weight + 0.01)
                    self.drm_weight = 1.0 - self.neural_weight
                elif drm_confidence > neural_confidence:
                    self.drm_weight = min(0.8, self.drm_weight + 0.01)
                    self.neural_weight = 1.0 - self.drm_weight

            self.session_stats['successful_predictions'] += 1
        else:
            self.session_stats['failed_predictions'] += 1
            print(f"  ❌ Nie rozpoznano instrukcji: {instruction}")

        # Zapisz do danych treningowych (oba systemy)
        self.training_data.append({
            'instruction': instruction,
            'type': instr_type,
            'semantic': semantic,
            'timestamp': self.drm.current_time,
            'neural_prediction': neural_prediction,
            'neural_confidence': neural_confidence
        })

        # Przygotuj dane dla treningu sieci neuronowej
        if instr_type and semantic:
            self.neural_training_data.append({
                'instruction': (instruction, instr_type),
                'target': self._create_neural_target(instr_type, semantic)
            })

    def _interpret_neural_output(self, output: np.ndarray) -> str:
        """Interpretuje wyjście sieci neuronowej"""
        # Mapowanie indeksów na typy instrukcji
        output_mapping = [
            'transfer_data', 'arithmetic_add', 'arithmetic_sub', 'arithmetic_mul',
            'arithmetic_div', 'compare_values', 'unconditional_jump', 'jump_if_equal',
            'jump_if_not_equal', 'stack_push', 'stack_pop', 'function_call',
            'function_return', 'memory_load', 'memory_store', 'no_operation'
        ]

        if len(output) >= len(output_mapping):
            max_idx = np.argmax(output[:len(output_mapping)])
            return output_mapping[max_idx]
        else:
            return "unknown_neural_output"

    def _create_neural_target(self, instr_type: str, semantic: str) -> np.ndarray:
        """Tworzy target vector dla treningu sieci neuronowej"""
        target = np.zeros(32)  # Output size sieci

        # Mapowanie typów na indeksy
        type_mapping = {
            'transfer_data': 0, 'arithmetic_add': 1, 'arithmetic_sub': 2,
            'arithmetic_mul': 3, 'arithmetic_div': 4, 'compare_values': 5,
            'unconditional_jump': 6, 'jump_if_equal': 7, 'jump_if_not_equal': 8,
            'stack_push': 9, 'stack_pop': 10, 'function_call': 11,
            'function_return': 12, 'memory_load': 13, 'memory_store': 14,
            'no_operation': 15
        }

        if instr_type in type_mapping:
            target[type_mapping[instr_type]] = 1.0

        # Dodaj informacje semantyczne do pozostałych wymiarów
        semantic_hash = hash(semantic) % 16
        target[16 + semantic_hash] = 0.5

        return target
    
    def simulate_processor(self, code: str):
        """Symuluje wykonanie kodu na procesorze"""
        self.processor.reset()
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        print("\n" + "="*60)
        print("🖥️  SYMULACJA PROCESORA")
        print("="*60)
        
        for line in lines:
            if line and not line.startswith(';'):
                success = self.processor.execute_instruction(line)
                if not success:
                    break
        
        # Pokaż stan końcowy
        state = self.processor.get_state()
        print(f"\n📊 STAN KOŃCOWY PROCESORA:")
        print(f"Rejestry: {state['registers']}")
        print(f"Flagi: {state['flags']}")
        print(f"Stos: {state['stack_size']} elementów")
        print(f"Program Counter: {state['program_counter']}")
        
        return self.processor.execution_log
    
    def introspect_code(self, code: str):
        """Introspekcja kodu z analizą semantyczną - NAPRAWIONA"""
        print("\n" + "="*60)
        print("🧠 INTROSPEKCJA KODU - ANALIZA SEMANTYCZNA")
        print("="*60)
        
        lines = code.strip().split('\n')
        analysis = []
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            instr_type, semantic = self.parser.parse_instruction(line)
            
            # NAPRAWKA: Lepsze dopasowywanie reguł
            matching_rule = None
            max_strength = 0
            
            for rule in self.drm.rules.values():
                # Sprawdź czy typ instrukcji pasuje do wzorca reguły
                if instr_type and (
                    instr_type.lower() in rule.pattern.lower() or 
                    rule.pattern.lower() in instr_type.lower() or
                    self._patterns_match(instr_type, rule.pattern)
                ):
                    strength = rule.calculate_strength(self.drm.current_time)
                    if strength > max_strength:
                        max_strength = strength
                        matching_rule = rule
            
            analysis_entry = {
                'line_num': i,
                'instruction': line,
                'type': instr_type or 'UNKNOWN',
                'semantic': semantic or 'nieznane znaczenie',
                'rule': matching_rule,
                'confidence': max_strength
            }
            analysis.append(analysis_entry)
        
        # Wyświetl analizę z POPRAWNYMI regułami
        for entry in analysis:
            print(f"\nLinia {entry['line_num']:2d}: {entry['instruction']}")
            print(f"  Typ: {entry['type']}")
            print(f"  Semantyka: {entry['semantic']}")
            if entry['rule']:
                print(f"  ✅ Reguła #{entry['rule'].id}: {entry['rule'].pattern}")
                print(f"     Siła: {entry['confidence']:.2f}")
                print(f"     Użycia: {entry['rule'].count}, Sukces: {entry['rule'].success_rate:.2f}")
            else:
                print(f"  ❌ Brak pasującej reguły")
        
        return analysis

    def _patterns_match(self, instr_type, rule_pattern):
        """Sprawdza czy wzorce się dopasowują"""
        # Mapowanie typów instrukcji na wzorce reguł
        type_to_pattern = {
            'transfer_data': ['MOV', 'LOAD', 'STORE'],
            'arithmetic_add': ['ADD'],
            'arithmetic_sub': ['SUB'],
            'compare_values': ['CMP'],
            'jump_if_equal': ['JE'],
            'stack_push': ['PUSH'],
            'stack_pop': ['POP'],
            'no_operation': ['NOP']
        }
        
        patterns = type_to_pattern.get(instr_type, [])
        return any(pattern.lower() in rule_pattern.lower() for pattern in patterns)
    
    def show_drm_status(self):
        """Wyświetla status Dynamic Rule Matrix"""
        print("\n" + "="*60)
        print("🧮 DYNAMIC RULE MATRIX - STATUS")
        print("="*60)
        
        strongest_rules = self.drm.get_strongest_rules(10)
        
        if not strongest_rules:
            print("Brak reguł w macierzy.")
            return
        
        for i, rule in enumerate(strongest_rules, 1):
            strength = rule.calculate_strength(self.drm.current_time)
            print(f"\nReguła #{rule.id:02d}: {rule.pattern} → \"{rule.semantic}\"")
            print(f"  Waga (W_i):      {rule.weight:.2f}")
            print(f"  Wystąpienia (C): {rule.count}")
            print(f"  Aktywność (U):   {rule.activity:.2f}")
            print(f"  Czas (T):        {self.drm.current_time} iteracji")
            print(f"  Korelacja (R):   {rule.correlation:.2f}")
            print(f"  Sukces:          {rule.success_rate:.2f}")
            print(f"  → Si = {rule.weight:.2f} * log({rule.count + 1}) * "
                  f"(1 + {rule.activity:.2f}/{self.drm.current_time}) * "
                  f"{rule.correlation:.2f} ≈ {strength:.2f}")
        
        print(f"\nCałkowita liczba reguł: {len(self.drm.rules)}")
        print(f"Aktualny czas systemu: {self.drm.current_time}")
    
    def drm_menu(self):
        """Menu zarządzania DRM"""
        while True:
            print("\n" + "─"*40)
            print("🧮 DYNAMIC RULE MATRIX - OPCJE")
            print("─"*40)
            print("[a] Dodaj nową regułę")
            print("[m] Modyfikuj regułę")
            print("[x] Usuń słabe reguły")
            print("[r] Raport zmian DRM")
            print("[s] Pokaż status DRM")
            print("[q] Wróć do menu głównego")
            
            choice = input("\nWybierz opcję: ").lower().strip()
            
            if choice == 'a':
                pattern = input("Wzorzec reguły: ")
                semantic = input("Znaczenie semantyczne: ")
                try:
                    weight = float(input("Waga początkowa (0.0-1.0): ") or "0.5")
                    rule_id = self.drm.add_rule(pattern, semantic, weight)
                    print(f"✅ Dodano regułę #{rule_id}")
                except ValueError:
                    print("❌ Błędna waga")
            
            elif choice == 'm':
                try:
                    rule_id = int(input("ID reguły do modyfikacji: "))
                    if rule_id in self.drm.rules:
                        rule = self.drm.rules[rule_id]
                        print(f"Aktualna reguła: {rule.pattern} → {rule.semantic}")
                        print(f"Aktualna waga: {rule.weight}")
                        
                        new_weight = input("Nowa waga (Enter = bez zmiany): ")
                        if new_weight:
                            rule.weight = max(0.0, min(1.0, float(new_weight)))
                        
                        new_semantic = input("Nowe znaczenie (Enter = bez zmiany): ")
                        if new_semantic:
                            rule.semantic = new_semantic
                        
                        print("✅ Reguła zaktualizowana")
                    else:
                        print("❌ Nie znaleziono reguły")
                except ValueError:
                    print("❌ Błędne ID")
            
            elif choice == 'x':
                removed = self.drm.remove_weak_rules(0.1)
                print(f"✅ Usunięto {removed} słabych reguł")
            
            elif choice == 'r':
                print(f"\n📊 RAPORT DRM:")
                print(f"Całkowita liczba reguł: {len(self.drm.rules)}")
                print(f"Czas systemu: {self.drm.current_time}")
                print(f"Współczynnik uczenia: {self.drm.learning_rate}")
                print(f"Współczynnik zaniku: {self.drm.decay_factor}")
            
            elif choice == 's':
                self.show_drm_status()
            
            elif choice == 'q':
                break
    
    def semantic_experiments(self):
        """Eksperymenty semantyczne"""
        print("\n" + "="*60)
        print("🔬 EKSPERYMENTY SEMANTYCZNE")
        print("="*60)
        
        experiments = [
            {
                'name': 'Analiza wzorców transferu danych',
                'code': 'MOV A, B\nMOV C, 100\nMOV D, A\nADD A, C',
                'focus': 'transfer_data'
            },
            {
                'name': 'Analiza operacji arytmetycznych',
                'code': 'ADD A, B\nSUB C, D\nMUL A, 5\nDIV C, 2',
                'focus': 'arithmetic'
            },
            {
                'name': 'Analiza kontroli przepływu',
                'code': 'CMP A, B\nJE label1\nJMP label2\nCALL func',
                'focus': 'control_flow'
            }
        ]
        
        for exp in experiments:
            print(f"\n🧪 {exp['name']}")
            print("─" * 40)
            print(f"Kod testowy:\n{exp['code']}")
            
            # Analizuj kod
            analysis = self.introspect_code(exp['code'])
            
            # Podsumowanie
            types_found = set(entry['type'] for entry in analysis)
            confidence_avg = sum(entry['confidence'] for entry in analysis) / len(analysis)
            
            print(f"\n📈 Podsumowanie eksperymentu:")
            print(f"Wykryte typy instrukcji: {', '.join(types_found)}")
            print(f"Średnia pewność rozpoznania: {confidence_avg:.2f}")
            
            input("\nNaciśnij Enter aby kontynuować...")
    
    def export_training_data(self):
        """Eksportuje dane treningowe"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Eksport danych treningowych
        training_file = f"logos_training_{timestamp}.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        
        # Eksport reguł DRM
        rules_data = {
            rule_id: asdict(rule) for rule_id, rule in self.drm.rules.items()
        }
        rules_file = f"logos_rules_{timestamp}.json"
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, ensure_ascii=False, indent=2)
        
        # Eksport statystyk
        stats_file = f"logos_stats_{timestamp}.json"
        stats = {
            **self.session_stats,
            'drm_rules_count': len(self.drm.rules),
            'drm_time': self.drm.current_time,
            'export_timestamp': timestamp
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Wyeksportowano dane:")
        print(f"  📄 Dane treningowe: {training_file}")
        print(f"  🧮 Reguły DRM: {rules_file}")
        print(f"  📊 Statystyki: {stats_file}")
    
    def show_training_dashboard(self):
        """Dashboard treningu w czasie rzeczywistym - HYBRYDOWY"""
        print("\n" + "="*80)
        print("📊 DASHBOARD TRENINGU LOGOS-ASM + DSNN")
        print("="*80)

        # Statystyki DRM
        total_rules = len(self.drm.rules)
        avg_strength = sum(rule.calculate_strength(self.drm.current_time)
                          for rule in self.drm.rules.values()) / max(total_rules, 1)

        print(f"🧮 DYNAMIC RULE MATRIX:")
        print(f"  Całkowita liczba reguł: {total_rules}")
        print(f"  Średnia siła: {avg_strength:.2f}")
        print(f"  Czas systemu: {self.drm.current_time}")

        # Statystyki sieci neuronowej
        neural_state = self.dsnn.get_network_state()
        print(f"\n🧠 DEEP SEMANTIC NEURAL NETWORK:")
        print(f"  Parametry: {neural_state['total_parameters']:,}")
        print(f"  Próbki treningowe: {neural_state['training_samples']}")
        if neural_state['current_loss'] > 0:
            print(f"  Aktualny loss: {neural_state['current_loss']:.4f}")
            print(f"  Dokładność: {neural_state['current_accuracy']:.2%}")

        # Tryb hybrydowy
        print(f"\n🤝 TRYB HYBRYDOWY:")
        print(f"  Status: {'✅ Włączony' if self.hybrid_mode else '❌ Wyłączony'}")
        if self.hybrid_mode:
            print(f"  Waga Neural: {self.neural_weight:.2f} | Waga DRM: {self.drm_weight:.2f}")

        # Top 5 najsilniejszych reguł
        strongest = self.drm.get_strongest_rules(5)
        print(f"\n🏆 TOP 5 NAJSILNIEJSZYCH REGUŁ DRM:")
        for i, rule in enumerate(strongest, 1):
            strength = rule.calculate_strength(self.drm.current_time)
            print(f"  {i}. {rule.pattern} → {rule.semantic[:30]}...")
            print(f"     Siła: {strength:.2f} | Użycia: {rule.count} | Sukces: {rule.success_rate:.2f}")

        # Wykres ASCII siły reguł
        print(f"\n📈 WYKRES SIŁY REGUŁ DRM:")
        self._draw_ascii_chart([rule.calculate_strength(self.drm.current_time)
                               for rule in strongest])

        # Statystyki sesji
        print(f"\n📋 STATYSTYKI SESJI:")
        for key, value in self.session_stats.items():
            display_key = key.replace('_', ' ').title()
            print(f"  {display_key}: {value}")

        # Accuracy rates
        total_predictions = (self.session_stats['successful_predictions'] +
                            self.session_stats['failed_predictions'])
        if total_predictions > 0:
            accuracy = self.session_stats['successful_predictions'] / total_predictions * 100
            print(f"\n📊 WYDAJNOŚĆ SYSTEMU:")
            print(f"  Ogólna dokładność: {accuracy:.1f}%")
            self._draw_progress_bar(accuracy, "Dokładność")

            # Hybrid performance
            if self.session_stats['hybrid_predictions'] > 0:
                hybrid_ratio = self.session_stats['hybrid_predictions'] / total_predictions * 100
                print(f"  Predykcje hybrydowe: {hybrid_ratio:.1f}%")
                self._draw_progress_bar(hybrid_ratio, "Tryb hybrydowy")

        # Neural network learning curves (jeśli dostępne)
        if len(self.dsnn.loss_history) > 0:
            print(f"\n🧠 KRZYWA UCZENIA SIECI NEURONOWEJ (ostatnie 10):")
            recent_losses = self.dsnn.loss_history[-10:]
            min_loss, max_loss = min(recent_losses), max(recent_losses)

            for i, loss in enumerate(recent_losses):
                if max_loss > min_loss:
                    normalized = (loss - min_loss) / (max_loss - min_loss)
                else:
                    normalized = 0.5
                bar_length = int((1 - normalized) * 20)  # Odwrócone - niższy loss = dłuższy pasek
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"  Epoka {i+1:2d}: |{bar}| Loss: {loss:.4f}")

        # Porównanie wydajności systemów
        if self.session_stats['neural_predictions'] > 0 and total_predictions > 0:
            neural_ratio = self.session_stats['neural_predictions'] / total_predictions * 100
            drm_ratio = 100 - neural_ratio

            print(f"\n⚖️ PODZIAŁ PREDYKCJI:")
            print(f"  DRM: {drm_ratio:.1f}% | Neural: {neural_ratio:.1f}%")

            if self.hybrid_mode:
                print(f"  Hybrydowe: {self.session_stats['hybrid_predictions']}")
                print(f"  Adaptacyjne wagi: Neural={self.neural_weight:.2f}, DRM={self.drm_weight:.2f}")

    def _draw_ascii_chart(self, values):
        """Rysuje prosty wykres ASCII"""
        if not values:
            return
        
        max_val = max(values)
        scale = 50 / max(max_val, 1)  # Skaluj do 50 znaków
        
        for i, val in enumerate(values, 1):
            bar_length = int(val * scale)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  #{i} |{bar}| {val:.2f}")

    def _draw_progress_bar(self, percentage, label):
        """Rysuje pasek postępu"""
        filled = int(percentage / 2)  # Skaluj do 50 znaków
        bar = "█" * filled + "░" * (50 - filled)
        print(f"  {label}: |{bar}| {percentage:.1f}%")
    
    def neural_network_menu(self):
        """Menu zarządzania siecią neuronową"""
        while True:
            print("\n" + "─"*50)
            print("🧠 DEEP SEMANTIC NEURAL NETWORK - MENU")
            print("─"*50)
            print("[1] Status sieci neuronowej")
            print("[2] Trening sieci na danych")
            print("[3] Test predykcji neuronowej")
            print("[4] Analiza podobieństwa semantycznego")
            print("[5] Analiza złożoności instrukcji")
            print("[6] Konfiguracja trybu hybrydowego")
            print("[7] Eksport modelu neuronowego")
            print("[8] Reset sieci neuronowej")
            print("[q] Wróć do menu głównego")

            choice = input("\nWybierz opcję: ").lower().strip()

            if choice == '1':
                self.show_neural_status()
            elif choice == '2':
                self.train_neural_network()
            elif choice == '3':
                self.test_neural_prediction()
            elif choice == '4':
                self.analyze_semantic_similarity()
            elif choice == '5':
                self.analyze_instruction_complexity()
            elif choice == '6':
                self.configure_hybrid_mode()
            elif choice == '7':
                self.export_neural_model()
            elif choice == '8':
                self.reset_neural_network()
            elif choice == 'q':
                break
            else:
                print("❌ Nieprawidłowa opcja")

    def show_neural_status(self):
        """Pokazuje status sieci neuronowej"""
        print("\n" + "="*60)
        print("🧠 STATUS DEEP SEMANTIC NEURAL NETWORK")
        print("="*60)

        state = self.dsnn.get_network_state()

        print(f"📊 ARCHITEKTURA SIECI:")
        print(f"  Rozmiary warstw: {state['layer_sizes']}")
        print(f"  Całkowita liczba parametrów: {state['total_parameters']:,}")
        print(f"  Rozmiar cache embeddings: {state['cache_size']}")

        print(f"\n📈 METRYKI TRENINGU:")
        print(f"  Próbek treningowych: {state['training_samples']}")
        print(f"  Aktualny loss: {state['current_loss']:.4f}")
        print(f"  Aktualna dokładność: {state['current_accuracy']:.2%}")

        print(f"\n🤝 TRYB HYBRYDOWY:")
        print(f"  Status: {'✅ Włączony' if self.hybrid_mode else '❌ Wyłączony'}")
        print(f"  Waga sieci neuronowej: {self.neural_weight:.2f}")
        print(f"  Waga DRM: {self.drm_weight:.2f}")

        # Pokaż ostatnie krzywe uczenia
        if len(self.dsnn.loss_history) > 0:
            print(f"\n📉 OSTATNIE 10 WARTOŚCI LOSS:")
            recent_losses = self.dsnn.loss_history[-10:]
            for i, loss in enumerate(recent_losses):
                print(f"  {i+1:2d}. {loss:.4f}")

    def train_neural_network(self):
        """Trenuje sieć neuronową na zebranych danych"""
        if not self.neural_training_data:
            print("❌ Brak danych treningowych dla sieci neuronowej")
            print("💡 Najpierw wytrenuj system na kodzie (opcja 1)")
            return

        print(f"\n🧠 TRENING SIECI NEURONOWEJ")
        print(f"Dostępne dane: {len(self.neural_training_data)} próbek")

        try:
            epochs = int(input("Liczba epok (domyślnie 10): ") or "10")
            batch_size = min(32, len(self.neural_training_data))

            print(f"\n🔄 Rozpoczynam trening ({epochs} epok, batch size: {batch_size})")

            # Przygotuj dane
            instructions = [item['instruction'] for item in self.neural_training_data]
            targets = [item['target'] for item in self.neural_training_data]

            # Trening w batch'ach
            for epoch in range(epochs):
                # Shuffle data
                indices = np.random.permutation(len(instructions))

                epoch_loss = 0
                batches = 0

                for i in range(0, len(instructions), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_instructions = [instructions[j] for j in batch_indices]
                    batch_targets = [targets[j] for j in batch_indices]

                    # Trening batch'a
                    self.dsnn.train_batch(batch_instructions, batch_targets, epochs=1)

                    if self.dsnn.loss_history:
                        epoch_loss += self.dsnn.loss_history[-1]
                        batches += 1

                avg_loss = epoch_loss / max(batches, 1)
                print(f"  Epoka {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

            print("✅ Trening zakończony!")

            # Aktualizuj krzywe uczenia
            if self.dsnn.loss_history:
                self.learning_curves['neural_loss'].extend(self.dsnn.loss_history[-epochs:])
            if self.dsnn.accuracy_history:
                self.learning_curves['neural_accuracy'].extend(self.dsnn.accuracy_history[-epochs:])

        except ValueError:
            print("❌ Błędna liczba epok")
        except Exception as e:
            print(f"❌ Błąd podczas treningu: {e}")

    def test_neural_prediction(self):
        """Testuje predykcję sieci neuronowej"""
        print("\n🧠 TEST PREDYKCJI NEURONOWEJ")
        instruction = input("Wprowadź instrukcję do analizy: ").strip()

        if not instruction:
            return

        try:
            # Analiza przez parser
            instr_type, semantic = self.parser.parse_instruction(instruction)

            # Predykcja neuronowa
            neural_output = self.dsnn.predict(instruction, instr_type)
            neural_prediction = self._interpret_neural_output(neural_output)
            confidence = float(np.max(neural_output))

            # Analiza złożoności
            complexity = self.dsnn.analyze_instruction_complexity(instruction, instr_type)

            print(f"\n📊 WYNIKI ANALIZY:")
            print(f"  Instrukcja: {instruction}")
            print(f"  Parser: {instr_type} → {semantic}")
            print(f"  Neural: {neural_prediction} (confidence: {confidence:.2f})")

            print(f"\n🔍 ANALIZA ZŁOŻONOŚCI:")
            for key, value in complexity.items():
                print(f"  {key.replace('_', ' ').title()}: {value:.3f}")

            # Pokaż top 5 neuronów wyjściowych
            print(f"\n🧠 TOP 5 AKTYWACJI WYJŚCIOWYCH:")
            top_indices = np.argsort(neural_output)[-5:][::-1]
            output_labels = ['transfer_data', 'arithmetic_add', 'arithmetic_sub', 'arithmetic_mul',
                           'arithmetic_div', 'compare_values', 'unconditional_jump', 'jump_if_equal']

            for i, idx in enumerate(top_indices):
                if idx < len(output_labels):
                    label = output_labels[idx]
                else:
                    label = f"output_{idx}"
                print(f"  {i+1}. {label}: {neural_output[idx]:.3f}")

        except Exception as e:
            print(f"❌ Błąd podczas predykcji: {e}")

    def analyze_semantic_similarity(self):
        """Analizuje podobieństwo semantyczne między instrukcjami"""
        print("\n🔍 ANALIZA PODOBIEŃSTWA SEMANTYCZNEGO")

        instr1 = input("Pierwsza instrukcja: ").strip()
        instr2 = input("Druga instrukcja: ").strip()

        if not instr1 or not instr2:
            return

        try:
            # Parsuj instrukcje
            type1, sem1 = self.parser.parse_instruction(instr1)
            type2, sem2 = self.parser.parse_instruction(instr2)

            # Oblicz podobieństwo
            similarity = self.dsnn.get_semantic_similarity(instr1, instr2, type1, type2)

            print(f"\n📊 WYNIKI ANALIZY:")
            print(f"  Instrukcja 1: {instr1} → {type1}")
            print(f"  Instrukcja 2: {instr2} → {type2}")
            print(f"  Podobieństwo semantyczne: {similarity:.3f}")

            if similarity > 0.8:
                print("  🟢 Bardzo wysokie podobieństwo")
            elif similarity > 0.6:
                print("  🟡 Wysokie podobieństwo")
            elif similarity > 0.4:
                print("  🟠 Średnie podobieństwo")
            else:
                print("  🔴 Niskie podobieństwo")

        except Exception as e:
            print(f"❌ Błąd podczas analizy: {e}")

    def analyze_instruction_complexity(self):
        """Analizuje złożoność instrukcji"""
        print("\n🔍 ANALIZA ZŁOŻONOŚCI INSTRUKCJI")
        instruction = input("Wprowadź instrukcję: ").strip()

        if not instruction:
            return

        try:
            instr_type, semantic = self.parser.parse_instruction(instruction)
            complexity = self.dsnn.analyze_instruction_complexity(instruction, instr_type)

            print(f"\n📊 ANALIZA ZŁOŻONOŚCI: {instruction}")
            print(f"  Typ: {instr_type}")
            print(f"  Semantyka: {semantic}")
            print(f"\n🔍 METRYKI ZŁOŻONOŚCI:")

            for key, value in complexity.items():
                if 'complexity' in key:
                    bar_length = int(value * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"  {key.replace('_', ' ').title():20s}: |{bar}| {value:.3f}")
                else:
                    print(f"  {key.replace('_', ' ').title():20s}: {value:.3f}")

        except Exception as e:
            print(f"❌ Błąd podczas analizy: {e}")

    def configure_hybrid_mode(self):
        """Konfiguruje tryb hybrydowy"""
        print("\n⚙️ KONFIGURACJA TRYBU HYBRYDOWEGO")
        print(f"Aktualny status: {'✅ Włączony' if self.hybrid_mode else '❌ Wyłączony'}")
        print(f"Waga sieci neuronowej: {self.neural_weight:.2f}")
        print(f"Waga DRM: {self.drm_weight:.2f}")

        print("\n[1] Włącz/wyłącz tryb hybrydowy")
        print("[2] Ustaw wagi ręcznie")
        print("[3] Reset do wartości domyślnych")
        print("[4] Tryb adaptacyjny (auto-tuning)")

        choice = input("Wybierz opcję [1-4]: ").strip()

        if choice == '1':
            self.hybrid_mode = not self.hybrid_mode
            status = "włączony" if self.hybrid_mode else "wyłączony"
            print(f"✅ Tryb hybrydowy {status}")

        elif choice == '2':
            try:
                neural_w = float(input(f"Waga sieci neuronowej (0.0-1.0, aktualnie {self.neural_weight:.2f}): "))
                if 0.0 <= neural_w <= 1.0:
                    self.neural_weight = neural_w
                    self.drm_weight = 1.0 - neural_w
                    print(f"✅ Ustawiono wagi: Neural={self.neural_weight:.2f}, DRM={self.drm_weight:.2f}")
                else:
                    print("❌ Waga musi być między 0.0 a 1.0")
            except ValueError:
                print("❌ Błędna wartość")

        elif choice == '3':
            self.neural_weight = 0.3
            self.drm_weight = 0.7
            self.hybrid_mode = True
            print("✅ Reset do wartości domyślnych")

        elif choice == '4':
            print("🤖 Tryb adaptacyjny włączony - wagi będą automatycznie dostrajane")
            self.hybrid_mode = True

    def export_neural_model(self):
        """Eksportuje model sieci neuronowej"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Eksport wag i biasów
            model_data = {
                'weights': [w.tolist() for w in self.dsnn.weights],
                'biases': [b.tolist() for b in self.dsnn.biases],
                'architecture': {
                    'input_size': self.dsnn.input_size,
                    'hidden_sizes': self.dsnn.hidden_sizes,
                    'output_size': self.dsnn.output_size
                },
                'training_history': {
                    'loss_history': self.dsnn.loss_history,
                    'accuracy_history': self.dsnn.accuracy_history
                },
                'hyperparameters': {
                    'learning_rate': self.dsnn.learning_rate,
                    'momentum': self.dsnn.momentum
                },
                'export_timestamp': timestamp
            }

            filename = f"logos_neural_model_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)

            print(f"✅ Model wyeksportowany: {filename}")
            print(f"📊 Parametry: {sum(w.size for w in self.dsnn.weights):,}")
            print(f"📈 Historia: {len(self.dsnn.loss_history)} epok")

        except Exception as e:
            print(f"❌ Błąd podczas eksportu: {e}")

    def reset_neural_network(self):
        """Resetuje sieć neuronową"""
        confirm = input("⚠️ Czy na pewno chcesz zresetować sieć neuronową? (tak/nie): ").lower()

        if confirm in ['tak', 'yes', 'y']:
            self.dsnn = DeepSemanticNeuralNetwork()
            self.neural_training_data = []
            self.learning_curves['neural_loss'] = []
            self.learning_curves['neural_accuracy'] = []
            print("✅ Sieć neuronowa została zresetowana")
        else:
            print("❌ Anulowano reset")

    def run(self):
        """Główna pętla programu z rozszerzonymi opcjami"""
        print("🧠 LOGOS-ASM — Terminal Świadomości Kodu")
        print("Semantyczny System Uczenia Kodu Maszynowego z DRM + DSNN")
        print("=" * 60)

        while True:
            print("\n[ MENU GŁÓWNE ]")
            print("1. Trening Modelu (ASCII / Binarne / ASM)")
            print("2. Symulacja Procesora")
            print("3. Introspekcja Kodu")
            print("4. Dynamic Rule Matrix (DRM)")
            print("5. 🧠 Deep Semantic Neural Network")
            print("6. Eksperymenty Semantyczne")
            print("7. 📊 Dashboard Treningu")
            print("8. 📈 Krzywe Uczenia")
            print("9. 🔍 Analiza Wzorców")
            print("10. 🎮 Interaktywny Trening")
            print("11. Zakończ i wyeksportuj dane")
            
            choice = input("\nWybierz opcję [1–11]: ").strip()

            if choice == '1':
                print("\nTRENING MODELU")
                print("Wprowadź kod (zakończ pustą linią):")
                code_lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    code_lines.append(line)
                
                if code_lines:
                    code = '\n'.join(code_lines)
                    
                    print("\nWybierz typ kodu:")
                    print("[1] Assembler")
                    print("[2] Kod binarny")
                    print("[3] ASCII/Tekst")
                    
                    code_type = input("Typ [1-3]: ").strip()
                    type_map = {'1': 'asm', '2': 'binary', '3': 'ascii'}
                    selected_type = type_map.get(code_type, 'asm')
                    
                    print(f"\n🔄 Trenowanie na kodzie typu: {selected_type}")
                    self.train_on_code(code, selected_type)
                    
                    print(f"✅ Przetworzono {len(code_lines)} linii kodu")
                    print(f"📊 Statystyki sesji:")
                    for key, value in self.session_stats.items():
                        print(f"  {key}: {value}")
            
            elif choice == '2':
                print("\nSYMULACJA PROCESORA")
                print("Wprowadź kod assemblera (zakończ pustą linią):")
                code_lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    code_lines.append(line)
                
                if code_lines:
                    code = '\n'.join(code_lines)
                    execution_log = self.simulate_processor(code)
                    
                    print(f"\n📝 LOG WYKONANIA ({len(execution_log)} kroków):")
                    for log_entry in execution_log[-10:]:  # Ostatnie 10 wpisów
                        print(f"  {log_entry}")
            
            elif choice == '3':
                print("\nINTROSPEKCJA KODU")
                print("Wprowadź kod do analizy (zakończ pustą linią):")
                code_lines = []
                while True:
                    line = input()
                    if not line:
                        break
                    code_lines.append(line)
                
                if code_lines:
                    code = '\n'.join(code_lines)
                    analysis = self.introspect_code(code)
                    
                    # Dodatkowa analiza świadomości
                    print(f"\n🧠 ANALIZA ŚWIADOMOŚCI KODU:")
                    semantic_patterns = {}
                    for entry in analysis:
                        sem_type = entry['type']
                        if sem_type in semantic_patterns:
                            semantic_patterns[sem_type] += 1
                        else:
                            semantic_patterns[sem_type] = 1
                    
                    print(f"Wykryte wzorce semantyczne:")
                    for pattern, count in semantic_patterns.items():
                        print(f"  {pattern}: {count} wystąpień")
                    
                    # Predykcja następnej instrukcji
                    if analysis:
                        last_type = analysis[-1]['type']
                        print(f"\n🔮 PREDYKCJA: Na podstawie '{last_type}' przewiduję:")
                        self._predict_next_instruction(last_type)
            
            elif choice == '4':
                self.drm_menu()

            elif choice == '5':
                self.neural_network_menu()

            elif choice == '6':
                self.semantic_experiments()

            elif choice == '7':
                self.show_training_dashboard()

            elif choice == '8':
                self.show_learning_curves()

            elif choice == '9':
                self.analyze_learning_patterns()

            elif choice == '10':
                self.interactive_training_mode()

            elif choice == '11':
                print("\n💾 EKSPORT DANYCH TRENINGOWYCH...")
                self.export_training_data()
                print("\n👋 Dziękuję za użycie LOGOS-ASM!")
                print("System zakończył pracę z pełną świadomością kodu.")
                break

            else:
                print("❌ Nieprawidłowa opcja. Wybierz 1-11.")
    
    def _predict_next_instruction(self, last_type: str):
        """Predykcja następnej instrukcji na podstawie wzorców"""
        predictions = {
            'transfer_data': ['arithmetic_add', 'compare_values', 'memory_store'],
            'arithmetic_add': ['compare_values', 'transfer_data', 'memory_store'],
            'compare_values': ['jump_if_equal', 'jump_if_not_equal', 'unconditional_jump'],
            'function_call': ['transfer_data', 'arithmetic_add', 'function_return'],
            'stack_push': ['stack_push', 'function_call', 'stack_pop'],
            'memory_load': ['transfer_data', 'arithmetic_add', 'compare_values']
        }
        
        possible = predictions.get(last_type, ['transfer_data'])
        
        # Znajdź najsilniejsze reguły dla predykcji
        best_predictions = []
        for pred_type in possible:
            for rule in self.drm.rules.values():
                if pred_type in rule.pattern.lower():
                    strength = rule.calculate_strength(self.drm.current_time)
                    best_predictions.append((pred_type, strength, rule.semantic))
        
        best_predictions.sort(key=lambda x: x[1], reverse=True)
        
        for i, (pred_type, strength, semantic) in enumerate(best_predictions[:3], 1):
            print(f"  {i}. {pred_type} (siła: {strength:.2f}) - {semantic}")

    def log_training_metrics(self):
        """Loguje metryki treningu"""
        timestamp = self.drm.current_time
        
        # Oblicz metryki
        total_rules = len(self.drm.rules)
        avg_strength = sum(rule.calculate_strength(timestamp) 
                          for rule in self.drm.rules.values()) / max(total_rules, 1)
        
        total_predictions = (self.session_stats['successful_predictions'] + 
                            self.session_stats['failed_predictions'])
        accuracy = (self.session_stats['successful_predictions'] / max(total_predictions, 1)) * 100
        
        # Zapisz metryki
        metrics = {
            'timestamp': timestamp,
            'total_rules': total_rules,
            'avg_strength': avg_strength,
            'accuracy': accuracy,
            'instructions_processed': self.session_stats['instructions_processed']
        }
        
        self.metrics_log.append(metrics)
        self.learning_curves['accuracy'].append(accuracy)
        self.learning_curves['rule_strength'].append(avg_strength)

    def show_learning_curves(self):
        """Pokazuje krzywe uczenia"""
        print("\n" + "="*80)
        print("📈 KRZYWE UCZENIA")
        print("="*80)
        
        if len(self.learning_curves['accuracy']) < 2:
            print("❌ Za mało danych do wyświetlenia krzywych")
            return
        
        # Wykres dokładności
        print("\n🎯 DOKŁADNOŚĆ W CZASIE:")
        self._plot_ascii_line(self.learning_curves['accuracy'], "Accuracy %")
        
        # Wykres siły reguł
        print("\n💪 ŚREDNIA SIŁA REGUŁ:")
        self._plot_ascii_line(self.learning_curves['rule_strength'], "Strength")
        
        # Trend analysis
        recent_accuracy = self.learning_curves['accuracy'][-5:]
        if len(recent_accuracy) >= 2:
            trend = "📈 Rosnący" if recent_accuracy[-1] > recent_accuracy[0] else "📉 Malejący"
            print(f"\n🔍 TREND (ostatnie 5 pomiarów): {trend}")

    def _plot_ascii_line(self, data, label):
        """Rysuje wykres liniowy ASCII"""
        if len(data) < 2:
            return
        
        # Normalizuj dane do 0-20 (wysokość wykresu)
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            normalized = [10] * len(data)
        else:
            normalized = [int((val - min_val) / (max_val - min_val) * 20) for val in data]
        
        # Rysuj wykres
        for row in range(20, -1, -1):
            line = f"{row:2d} |"
            for val in normalized:
                if val >= row:
                    line += "█"
                else:
                    line += " "
            if row == 20:
                line += f" Max: {max_val:.2f}"
            elif row == 0:
                line += f" Min: {min_val:.2f}"
            print(line)
        
        # Oś X
        x_axis = "   +" + "─" * len(data)
        print(x_axis)
        print(f"   {label} over time")

    def analyze_learning_patterns(self):
        """Analizuje wzorce uczenia się systemu"""
        print("\n" + "="*80)
        print("🔍 ANALIZA WZORCÓW UCZENIA")
        print("="*80)
        
        # Analiza reguł według kategorii
        categories = {}
        for rule in self.drm.rules.values():
            category = self._categorize_rule(rule.pattern)
            if category not in categories:
                categories[category] = []
            categories[category].append(rule)
        
        print("\n📊 REGUŁY WEDŁUG KATEGORII:")
        for category, rules in categories.items():
            avg_strength = sum(rule.calculate_strength(self.drm.current_time) 
                              for rule in rules) / len(rules)
            print(f"  {category}: {len(rules)} reguł, średnia siła: {avg_strength:.2f}")
        
        # Analiza ewolucji reguł
        print("\n🧬 EWOLUCJA NAJSILNIEJSZYCH REGUŁ:")
        strongest = self.drm.get_strongest_rules(3)
        for rule in strongest:
            print(f"\nReguła #{rule.id}: {rule.pattern}")
            print(f"  Utworzona: iteracja {rule.time_created}")
            print(f"  Wiek: {self.drm.current_time - rule.time_created} iteracji")
            print(f"  Wzrost siły: {self._calculate_strength_growth(rule):.2f}/iteracja")
            print(f"  Efektywność: {rule.success_rate:.2f}")

    def _categorize_rule(self, pattern):
        """Kategoryzuje regułę według wzorca"""
        pattern_lower = pattern.lower()
        if any(x in pattern_lower for x in ['mov', 'load', 'store']):
            return "Transfer danych"
        elif any(x in pattern_lower for x in ['add', 'sub', 'mul', 'div']):
            return "Arytmetyka"
        elif any(x in pattern_lower for x in ['cmp', 'test']):
            return "Porównania"
        elif any(x in pattern_lower for x in ['jmp', 'je', 'jne', 'call']):
            return "Kontrola przepływu"
        elif any(x in pattern_lower for x in ['push', 'pop']):
            return "Stos"
        else:
            return "Inne"

    def _calculate_strength_growth(self, rule):
        """Oblicza wzrost siły reguły na iterację"""
        age = max(self.drm.current_time - rule.time_created, 1)
        current_strength = rule.calculate_strength(self.drm.current_time)
        return current_strength / age

    def interactive_training_mode(self):
        """Interaktywny tryb treningu z live feedback"""
        print("\n" + "="*80)
        print("🎮 INTERAKTYWNY TRYB TRENINGU")
        print("="*80)
        print("Wprowadzaj kod linia po linii. Wpisz 'help' dla pomocy, 'quit' aby wyjść.")
        
        while True:
            print(f"\n[Iteracja {self.drm.current_time}] Wprowadź instrukcję:")
            instruction = input(">>> ").strip()
            
            if instruction.lower() == 'quit':
                break
            elif instruction.lower() == 'help':
                self._show_training_help()
                continue
            elif instruction.lower() == 'stats':
                self.show_training_dashboard()
                continue
            elif instruction.lower() == 'predict':
                self._interactive_prediction()
                continue
            elif not instruction:
                continue
            
            # Przetwórz instrukcję
            print(f"\n🔄 Przetwarzanie: {instruction}")
            
            # Pokaż predykcję przed przetworzeniem
            predicted_type = self._predict_instruction_type(instruction)
            print(f"🔮 Predykcja: {predicted_type}")
            
            # Przetwórz
            old_stats = self.session_stats.copy()
            self._process_instruction(instruction)
            
            # Pokaż rezultat
            instr_type, semantic = self.parser.parse_instruction(instruction)
            correct = predicted_type == instr_type
            
            print(f"✅ Rzeczywisty typ: {instr_type}")
            print(f"📝 Semantyka: {semantic}")
            print(f"🎯 Predykcja {'✓ POPRAWNA' if correct else '✗ BŁĘDNA'}")
            
            # Pokaż zmiany w regułach
            self._show_rule_changes(instruction, instr_type)
            
            # Aktualizuj metryki
            self.log_training_metrics()
            self.drm.tick()

    def _predict_instruction_type(self, instruction):
        """Przewiduje typ instrukcji przed przetworzeniem"""
        # Znajdź najbardziej pasującą regułę
        best_match = None
        best_strength = 0
        
        for rule in self.drm.rules.values():
            if rule.pattern.upper() in instruction.upper():
                strength = rule.calculate_strength(self.drm.current_time)
                if strength > best_strength:
                    best_strength = strength
                    best_match = rule
        
        return best_match.pattern if best_match else "UNKNOWN"

    def _show_rule_changes(self, instruction, instr_type):
        """Pokazuje zmiany w regułach po przetworzeniu"""
        print(f"\n📊 ZMIANY W REGUŁACH:")
        
        # Znajdź regułę która została zaktualizowana
        for rule in self.drm.rules.values():
            if instr_type and instr_type.upper() in rule.pattern.upper():
                strength = rule.calculate_strength(self.drm.current_time)
                print(f"  Reguła #{rule.id}: {rule.pattern}")
                print(f"    Siła: {strength:.2f}")
                print(f"    Użycia: {rule.count}")
                print(f"    Sukces: {rule.success_rate:.2f}")
                break

    def _interactive_prediction(self):
        """Interaktywna predykcja następnej instrukcji"""
        print("\n🔮 PREDYKCJA NASTĘPNEJ INSTRUKCJI")
        last_instruction = input("Podaj ostatnią instrukcję: ").strip()
        
        if last_instruction:
            instr_type, _ = self.parser.parse_instruction(last_instruction)
            if instr_type:
                print(f"\nNa podstawie '{instr_type}' przewiduję:")
                self._predict_next_instruction(instr_type)
            else:
                print("❌ Nie rozpoznano typu instrukcji")

    def debug_rules(self):
        """Debug mode - pokaż wszystkie reguły"""
        print("\n" + "="*60)
        print("🔍 DEBUG - WSZYSTKIE REGUŁY")
        print("="*60)
        
        if not self.drm.rules:
            print("❌ Brak reguł w systemie!")
            return
        
        for rule_id, rule in self.drm.rules.items():
            strength = rule.calculate_strength(self.drm.current_time)
            print(f"\nReguła #{rule_id}:")
            print(f"  Wzorzec: '{rule.pattern}'")
            print(f"  Semantyka: '{rule.semantic}'")
            print(f"  Siła: {strength:.2f}")
            print(f"  Waga: {rule.weight:.2f}")
            print(f"  Użycia: {rule.count}")
            print(f"  Sukces: {rule.success_rate:.2f}")
            print(f"  Aktywność: {rule.activity:.2f}")

def main():
    """Funkcja główna programu"""
    try:
        import os
        import sys
        import json
        import math
        import re
        from datetime import datetime
        from dataclasses import dataclass, asdict
        from typing import Dict, List, Optional, Tuple
        
        # Sprawdź czy wszystkie moduły są dostępne
        print("🔧 Inicjalizacja LOGOS-ASM...")
        
        # Utwórz instancję systemu
        logos = LogosASM()
        
        # Uruchom główną pętlę
        logos.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Przerwano przez użytkownika")
        print("💾 Zapisywanie stanu systemu...")
        
        # Awaryjny zapis danych
        try:
            logos.export_training_data()
            print("✅ Stan systemu zapisany pomyślnie")
        except:
            print("❌ Błąd podczas zapisu stanu")
        
        print("👋 LOGOS-ASM zakończony")
        
    except Exception as e:
        print(f"\n💥 BŁĄD KRYTYCZNY: {e}")
        print("🔍 Sprawdź logi systemu i spróbuj ponownie")
        sys.exit(1)


if __name__ == "__main__":
    # Banner startowy
    print("=" * 60)
    print("🧠 LOGOS-ASM — Terminal Świadomości Kodu")
    print("Semantyczny System Uczenia Kodu Maszynowego z DRM")
    print("=" * 60)
    print("📋 ASCII | Binarne | Asembler | Dynamic Rule Matrix")
    print("🎯 Cel: Semantyczne zrozumienie kodu maszynowego")
    print("⚡ Status: Gotowy do nauki i analizy")
    print("=" * 60)
    
    main()
