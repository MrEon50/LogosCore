#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Quick Training Script dla LOGOS-ASM
Szybkie trenowanie na różnych typach kodu
"""

from logoscore import LogosASM
import time

def train_on_samples():
    """Trenuje system na różnych próbkach kodu"""
    logos = LogosASM()
    
    # 1. Kod arytmetyczny
    arithmetic_code = """
    MOV EAX, 10
    MOV EBX, 20
    ADD EAX, EBX
    SUB EAX, 5
    MUL EAX, 2
    DIV EAX, 3
    CMP EAX, 50
    """
    
    # 2. Kod funkcji
    function_code = """
    PUSH EBP
    MOV EBP, ESP
    SUB ESP, 16
    MOV [EBP-4], EAX
    CALL helper_func
    ADD ESP, 16
    POP EBP
    RET
    """
    
    # 3. Kod pętli
    loop_code = """
    MOV ECX, 10
    loop_start:
    PUSH ECX
    CALL process_item
    POP ECX
    DEC ECX
    CMP ECX, 0
    JNE loop_start
    """
    
    # 4. Kod szyfrowania (XOR)
    crypto_code = """
    MOV ESI, data_ptr
    MOV EDI, key_ptr
    MOV ECX, data_len
    xor_loop:
    MOV AL, [ESI]
    XOR AL, [EDI]
    MOV [ESI], AL
    INC ESI
    INC EDI
    LOOP xor_loop
    """
    
    # 5. Kod game hackingu
    game_code = """
    MOV EAX, player_ptr
    MOV EBX, [EAX+health_offset]
    CMP EBX, 0
    JLE player_dead
    ADD EBX, 100
    MOV [EAX+health_offset], EBX
    """
    
    samples = [
        ("Arytmetyka", arithmetic_code),
        ("Funkcje", function_code), 
        ("Pętle", loop_code),
        ("Szyfrowanie", crypto_code),
        ("Game Hacking", game_code)
    ]
    
    print("🧠 SZYBKI TRENING LOGOS-ASM")
    print("=" * 50)
    
    for name, code in samples:
        print(f"\n📚 Trenowanie: {name}")
        print("-" * 30)
        
        # Trening bez debugów
        start_time = time.time()
        logos.train_on_code(code, 'asm')
        end_time = time.time()
        
        print(f"⏱️ Czas: {end_time - start_time:.2f}s")
        print(f"📊 Reguły: {len(logos.drm.rules)}")
        print(f"🧠 Dane neural: {len(logos.neural_training_data)}")
    
    # Pokaż końcowe statystyki
    print(f"\n🎯 KOŃCOWE STATYSTYKI:")
    print(f"📊 Całkowite reguły DRM: {len(logos.drm.rules)}")
    print(f"🧠 Próbki neuronowe: {len(logos.neural_training_data)}")
    print(f"✅ Przetworzonych instrukcji: {logos.session_stats['instructions_processed']}")
    print(f"🎯 Dokładność: {logos.session_stats['successful_predictions']}/{logos.session_stats['instructions_processed']}")
    
    return logos

def test_predictions(logos):
    """Testuje predykcje na nowych instrukcjach"""
    print(f"\n🔮 TEST PREDYKCJI")
    print("=" * 50)
    
    test_instructions = [
        "MOV EAX, [ESP+4]",  # Dostęp do argumentu funkcji
        "XOR EAX, EAX",      # Zerowanie rejestru
        "PUSH 0x41414141",   # Buffer overflow pattern
        "CALL GetProcAddress", # WinAPI call
        "JMP short +0x10",   # Skok względny
        "LEA EAX, [EBP-8]",  # Load effective address
    ]
    
    for instr in test_instructions:
        print(f"\n🔍 Instrukcja: {instr}")
        
        # Parser analysis
        instr_type, semantic = logos.parser.parse_instruction(instr)
        print(f"  📝 Parser: {instr_type} → {semantic}")
        
        # Neural prediction
        try:
            neural_output = logos.dsnn.predict(instr, instr_type)
            neural_pred = logos._interpret_neural_output(neural_output)
            neural_conf = float(max(neural_output))
            print(f"  🧠 Neural: {neural_pred} (conf: {neural_conf:.3f})")
        except:
            print(f"  🧠 Neural: Błąd predykcji")
        
        # Complexity analysis
        try:
            complexity = logos.dsnn.analyze_instruction_complexity(instr, instr_type)
            print(f"  📊 Złożoność: {complexity['total_complexity']:.2f}")
        except:
            print(f"  📊 Złożoność: Błąd analizy")

def analyze_malware_sample():
    """Przykład analizy próbki malware"""
    print(f"\n🦠 ANALIZA PRÓBKI MALWARE")
    print("=" * 50)
    
    # Symulacja kodu malware
    malware_code = """
    ; Typical malware entry point
    CALL get_delta
    get_delta:
    POP EBP
    SUB EBP, offset get_delta
    
    ; API resolution
    MOV EAX, [EBP + kernel32_hash]
    PUSH EAX
    CALL find_api_by_hash
    
    ; String decryption
    LEA ESI, [EBP + encrypted_string]
    MOV ECX, string_length
    decrypt_loop:
    XOR BYTE PTR [ESI], 0x42
    INC ESI
    LOOP decrypt_loop
    
    ; Payload execution
    PUSH 0
    PUSH encrypted_string
    PUSH 0
    CALL CreateProcessA
    """
    
    logos = LogosASM()
    logos.train_on_code(malware_code, 'asm')
    
    print("🔍 Wykryte wzorce:")
    strongest_rules = logos.drm.get_strongest_rules(5)
    for i, rule in enumerate(strongest_rules, 1):
        strength = rule.calculate_strength(logos.drm.current_time)
        if strength > 0:
            print(f"  {i}. {rule.pattern} (siła: {strength:.2f}) - {rule.semantic}")
    
    # Analiza podejrzanych wzorców
    suspicious_patterns = [
        "XOR BYTE PTR",  # Szyfrowanie
        "CALL find_api", # API hashing
        "POP EBP",       # Delta offset
        "CreateProcessA" # Process creation
    ]
    
    print(f"\n⚠️ Podejrzane wzorce:")
    for pattern in suspicious_patterns:
        if any(pattern.lower() in line.lower() for line in malware_code.split('\n')):
            print(f"  🚨 Znaleziono: {pattern}")

def game_hacking_analysis():
    """Przykład analizy do game hackingu"""
    print(f"\n🎮 ANALIZA GAME HACKING")
    print("=" * 50)
    
    # Kod modyfikujący zdrowie gracza
    game_hack_code = """
    ; Find player object
    MOV EAX, [player_base]
    TEST EAX, EAX
    JZ exit
    
    ; Modify health
    MOV EBX, [EAX + 0x4C]  ; health offset
    CMP EBX, max_health
    JGE skip_heal
    MOV [EAX + 0x4C], max_health
    
    skip_heal:
    ; Modify mana
    MOV ECX, [EAX + 0x50]  ; mana offset  
    MOV [EAX + 0x50], max_mana
    
    ; God mode
    MOV BYTE PTR [EAX + 0x60], 1
    
    exit:
    RET
    """
    
    logos = LogosASM()
    logos.train_on_code(game_hack_code, 'asm')
    
    print("🎯 Wykryte wzorce modyfikacji:")
    for entry in logos.training_data:
        if 'MOV' in entry['instruction'] and '[' in entry['instruction']:
            print(f"  📝 {entry['instruction']} - {entry['semantic']}")

def main():
    """Główna funkcja demonstracyjna"""
    print("🚀 LOGOS-ASM - SZYBKIE WYKORZYSTANIE")
    print("=" * 60)
    
    # 1. Podstawowy trening
    logos = train_on_samples()
    
    # 2. Test predykcji
    test_predictions(logos)
    
    # 3. Analiza malware
    analyze_malware_sample()
    
    # 4. Game hacking
    game_hacking_analysis()
    
    print(f"\n🎉 GOTOWE!")
    print("💡 Uruchom 'python logoscore.py' dla pełnego interfejsu")

if __name__ == "__main__":
    main()
