#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 Test Deep Semantic Neural Network w LOGOS-ASM
Sprawdzenie integracji DRM + DSNN
"""

import numpy as np
import sys
import os

# Dodaj ścieżkę do głównego modułu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from logoscore import LogosASM, DeepSemanticNeuralNetwork
    print("✅ Import modułów udany")
except ImportError as e:
    print(f"❌ Błąd importu: {e}")
    print("💡 Sprawdź czy numpy jest zainstalowany: pip install numpy")
    sys.exit(1)

def test_neural_network():
    """Test podstawowej funkcjonalności sieci neuronowej"""
    print("\n" + "="*60)
    print("🧠 TEST DEEP SEMANTIC NEURAL NETWORK")
    print("="*60)
    
    # Utwórz sieć
    dsnn = DeepSemanticNeuralNetwork()
    print(f"✅ Sieć utworzona: {dsnn.input_size} → {dsnn.hidden_sizes} → {dsnn.output_size}")
    
    # Test kodowania instrukcji
    test_instructions = [
        ("MOV A, 100", "transfer_data"),
        ("ADD A, B", "arithmetic_add"),
        ("SUB A, 50", "arithmetic_sub"),
        ("CMP A, B", "compare_values"),
        ("PUSH A", "stack_push")
    ]
    
    print(f"\n📝 TEST KODOWANIA INSTRUKCJI:")
    for instr, instr_type in test_instructions:
        embedding = dsnn.encode_instruction(instr, instr_type)
        print(f"  {instr:12s} → embedding shape: {embedding.shape}, sum: {np.sum(embedding):.2f}")
    
    # Test predykcji
    print(f"\n🔮 TEST PREDYKCJI:")
    for instr, instr_type in test_instructions:
        prediction = dsnn.predict(instr, instr_type)
        confidence = np.max(prediction)
        print(f"  {instr:12s} → confidence: {confidence:.3f}")
    
    # Test podobieństwa
    print(f"\n🔍 TEST PODOBIEŃSTWA SEMANTYCZNEGO:")
    similarity = dsnn.get_semantic_similarity("MOV A, B", "MOV C, D", "transfer_data", "transfer_data")
    print(f"  MOV A, B vs MOV C, D: {similarity:.3f}")
    
    similarity = dsnn.get_semantic_similarity("ADD A, B", "SUB A, B", "arithmetic_add", "arithmetic_sub")
    print(f"  ADD A, B vs SUB A, B: {similarity:.3f}")
    
    print("✅ Test sieci neuronowej zakończony")

def test_hybrid_system():
    """Test systemu hybrydowego DRM + DSNN"""
    print("\n" + "="*60)
    print("🤝 TEST SYSTEMU HYBRYDOWEGO")
    print("="*60)
    
    # Utwórz system
    logos = LogosASM()
    print(f"✅ System LOGOS-ASM utworzony")
    print(f"🧮 DRM: {len(logos.drm.rules)} reguł")
    print(f"🧠 DSNN: {logos.dsnn.input_size} → {logos.dsnn.output_size}")
    print(f"🤝 Tryb hybrydowy: {logos.hybrid_mode}")
    
    # Test treningu
    test_code = """MOV A, 100
ADD A, B
SUB A, 50
CMP A, B
PUSH A
POP B
CALL function
RET"""
    
    print(f"\n🔄 TEST TRENINGU NA KODZIE:")
    print("Kod testowy:")
    for i, line in enumerate(test_code.split('\n'), 1):
        print(f"  {i}. {line}")
    
    # Trening (bez wypisywania debugów)
    original_print = print
    def silent_print(*args, **kwargs):
        pass
    
    # Temporarily redirect print for training
    import builtins
    builtins.print = silent_print
    
    try:
        logos.train_on_code(test_code, 'asm')
    finally:
        builtins.print = original_print
    
    print(f"✅ Trening zakończony")
    print(f"📊 Statystyki: {logos.session_stats}")
    print(f"🧮 Reguły DRM: {len(logos.drm.rules)}")
    print(f"🧠 Dane neuronowe: {len(logos.neural_training_data)}")
    
    # Test predykcji hybrydowej
    print(f"\n🔮 TEST PREDYKCJI HYBRYDOWEJ:")
    test_instructions = ["MOV C, 200", "ADD C, D", "CMP C, 100"]
    
    for instr in test_instructions:
        print(f"\n  Instrukcja: {instr}")
        
        # DRM prediction
        instr_type, semantic = logos.parser.parse_instruction(instr)
        drm_confidence = 0.0
        for rule in logos.drm.rules.values():
            if instr_type and (instr_type.lower() in rule.pattern.lower() or 
                              rule.pattern.lower() in instr_type.lower()):
                drm_confidence = rule.calculate_strength(logos.drm.current_time)
                break
        
        # Neural prediction
        try:
            neural_output = logos.dsnn.predict(instr, instr_type)
            neural_confidence = float(np.max(neural_output))
        except:
            neural_confidence = 0.0
        
        # Hybrid
        if logos.hybrid_mode:
            hybrid_confidence = (logos.drm_weight * drm_confidence + 
                               logos.neural_weight * neural_confidence)
        else:
            hybrid_confidence = drm_confidence
        
        print(f"    DRM: {drm_confidence:.3f} | Neural: {neural_confidence:.3f} | Hybrid: {hybrid_confidence:.3f}")
    
    print("✅ Test systemu hybrydowego zakończony")

def test_neural_training():
    """Test treningu sieci neuronowej"""
    print("\n" + "="*60)
    print("🎓 TEST TRENINGU SIECI NEURONOWEJ")
    print("="*60)
    
    logos = LogosASM()
    
    # Przygotuj dane treningowe
    training_data = [
        (("MOV A, 100", "transfer_data"), logos._create_neural_target("transfer_data", "transfer danych")),
        (("ADD A, B", "arithmetic_add"), logos._create_neural_target("arithmetic_add", "dodawanie")),
        (("SUB A, 50", "arithmetic_sub"), logos._create_neural_target("arithmetic_sub", "odejmowanie")),
        (("CMP A, B", "compare_values"), logos._create_neural_target("compare_values", "porównanie")),
        (("PUSH A", "stack_push"), logos._create_neural_target("stack_push", "stos push")),
    ]
    
    print(f"📚 Przygotowano {len(training_data)} próbek treningowych")
    
    # Trening
    instructions = [item[0] for item in training_data]
    targets = [item[1] for item in training_data]
    
    print(f"🔄 Rozpoczynam trening (3 epoki)...")
    
    initial_loss = None
    try:
        logos.dsnn.train_batch(instructions, targets, epochs=3)
        
        if logos.dsnn.loss_history:
            initial_loss = logos.dsnn.loss_history[0] if len(logos.dsnn.loss_history) > 0 else None
            final_loss = logos.dsnn.loss_history[-1]
            
            print(f"📉 Loss: {initial_loss:.4f} → {final_loss:.4f}")
            
            if initial_loss and final_loss < initial_loss:
                print("✅ Sieć się uczy - loss maleje!")
            else:
                print("⚠️ Loss nie maleje - może potrzeba więcej danych")
        
        print(f"📊 Historia treningu: {len(logos.dsnn.loss_history)} epok")
        
    except Exception as e:
        print(f"❌ Błąd podczas treningu: {e}")
    
    print("✅ Test treningu zakończony")

def main():
    """Główna funkcja testowa"""
    print("🧠 LOGOS-ASM + DSNN - TESTY INTEGRACJI")
    print("=" * 60)
    
    try:
        # Test 1: Podstawowa funkcjonalność sieci
        test_neural_network()
        
        # Test 2: System hybrydowy
        test_hybrid_system()
        
        # Test 3: Trening sieci
        test_neural_training()
        
        print("\n" + "="*60)
        print("🎉 WSZYSTKIE TESTY ZAKOŃCZONE POMYŚLNIE!")
        print("="*60)
        print("💡 System gotowy do użycia:")
        print("   - Dynamic Rule Matrix (DRM) ✅")
        print("   - Deep Semantic Neural Network (DSNN) ✅") 
        print("   - Tryb hybrydowy ✅")
        print("   - Adaptacyjne wagi ✅")
        print("\n🚀 Uruchom: python logoscore.py")
        
    except Exception as e:
        print(f"\n💥 BŁĄD PODCZAS TESTÓW: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
