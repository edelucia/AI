####################################################################
# Questo script è uno strumento di Deep Binary Diffing             #
# progettato per l'analisi forense di modelli AI in formato GGUF.  #
# Questo tool esegue un confronto bit-per-bit tra un modello       #
# "Golden Image" e un artefatto sospetto, utilizzando un approccio #
# ibrido (Parser + Raw Stream).                                    #
#                                                                  #
#  Utile per:                                                      #
#                                                                  #
#      ** Metadata Audit                                           #
#      ** Lobotomy Attacks Detection                               #
#      ** Noise Injection Detection                                #
#      ** AI Swap Attack Detection                                 #
#                                                                  #
#  Author:  Emanuele De Lucia                                      #
#  Contact: https://www.linkedin.com/in/emanuele-de-lucia/         # 
#                                                                  #
####################################################################

import gguf
import numpy as np
import sys
import os
import struct

class MasterForensics:
    def __init__(self, clean_path, suspect_path):
        self.clean_path = clean_path
        self.suspect_path = suspect_path
        
        if not os.path.exists(clean_path) or not os.path.exists(suspect_path):
            print("[-] Errore: File non trovati.")
            sys.exit(1)

        print(f"[*] MASTER FORENSICS TOOL INITIALIZED")
        print(f"    REF:  {os.path.basename(clean_path)}")
        print(f"    SUSP: {os.path.basename(suspect_path)}")

        try:
            self.r_clean = gguf.GGUFReader(clean_path, mode='r')
            self.r_susp = gguf.GGUFReader(suspect_path, mode='r')
            print("[+] Header GGUF validi.")
        except Exception as e:
            print(f"[!] Errore critico nel parsing GGUF: {e}")
            print("    Il file potrebbe essere corrotto strutturalmente.")
            sys.exit(1)

        self.f_clean_raw = open(clean_path, 'rb')
        self.f_susp_raw = open(suspect_path, 'rb')

    def _safe_compare(self, v1, v2):
        if isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        return v1 == v2

    def audit_metadata(self):
        print(f"\n=== [1] METADATA AUDIT ===")
        keys_c = set(self.r_clean.fields.keys())
        keys_s = set(self.r_susp.fields.keys())
        
        diffs = 0
        for k in keys_c | keys_s:
            if k not in keys_c:
                print(f"[!] Chiave AGGIUNTA: {k}")
                diffs += 1
                continue
            if k not in keys_s:
                print(f"[!] Chiave RIMOSSA: {k}")
                diffs += 1
                continue
            
            val_c = self.r_clean.fields[k].data
            val_s = self.r_susp.fields[k].data
            
            if not self._safe_compare(val_c, val_s):
                print(f"[!] MODIFICA in {k}:")
                vc_str = str(val_c)
                vs_str = str(val_s)
                print(f"    Orig: {vc_str[:50]}{'...' if len(vc_str)>50 else ''}")
                print(f"    Susp: {vs_str[:50]}{'...' if len(vs_str)>50 else ''}")
                diffs += 1
        
        if diffs == 0:
            print("[OK] Metadati Identici.")

    def audit_tensors(self):
        print(f"\n=== [2] TENSOR WEIGHT AUDIT ===")
        tensors_c = {t.name: t for t in self.r_clean.tensors}
        tensors_s = {t.name: t for t in self.r_susp.tensors}
        
        diffs = []
        for name in set(tensors_c.keys()) & set(tensors_s.keys()):
            t_c = tensors_c[name]
            t_s = tensors_s[name]
            
            if tuple(t_c.shape) != tuple(t_s.shape):
                print(f"[!] CAMBIO DIMENSIONE {name}: {t_c.shape} -> {t_s.shape}")
                continue
            
            if not np.array_equal(t_c.data, t_s.data):
                mismatch = np.sum(t_c.data != t_s.data)
                total = t_c.data.nbytes
                pct = (mismatch / total) * 100
                diffs.append((name, pct))

        if not diffs:
            print("[OK] Pesi Neurali Identici (Nessuna Lobotomia/Noise rilevata).")
        else:
            print(f"[!!!] RILEVATE MODIFICHE AI PESI SU {len(diffs)} TENSORI:")
            for name, pct in diffs:
                print(f"    >> {name:<30} | {pct:.6f}% bytes modificati")

    def _find_vocab_start_raw(self, f_handle, hint_size=32000):
        """Cerca il Magic Number dell'array vocabolario nei primi 50MB"""
        f_handle.seek(0)
        buffer = f_handle.read(50 * 1024 * 1024)
        
        magic_pattern = struct.pack('<IIQ', 8, 9, hint_size)
        offset = buffer.find(magic_pattern)
        
        if offset != -1: return offset + 16
        
        offset = buffer.find(struct.pack('<Q', hint_size))
        if offset != -1: return offset + 8
        
        return -1

    def audit_vocabulary_raw(self):
        print(f"\n=== [3] RAW VOCABULARY AUDIT ===")
        
        try:
            t = next(x for x in self.r_clean.tensors if 'token_embd' in x.name)
            vocab_size = max(t.shape)
        except:
            vocab_size = 32000
            
        print(f"[*] Vocabolario target: {vocab_size} tokens")

        offset_c = self._find_vocab_start_raw(self.f_clean_raw, vocab_size)
        offset_s = self._find_vocab_start_raw(self.f_susp_raw, vocab_size)
        
        if offset_c == -1 or offset_s == -1:
            print("[!] IMPOSSIBILE ALLINEARE GLI ARRAY VOCABOLARIO.")
            print("    Il file potrebbe avere una struttura header molto diversa.")
            return

        print(f"[*] Allineamento OK. Clean@{offset_c} | Susp@{offset_s}")
        
        self.f_clean_raw.seek(offset_c)
        self.f_susp_raw.seek(offset_s)
        
        anomalies = []
        
        for i in range(vocab_size):
            b_len_c = self.f_clean_raw.read(8)
            b_len_s = self.f_susp_raw.read(8)
            
            if not b_len_c or not b_len_s: break
            
            len_c = struct.unpack("<Q", b_len_c)[0]
            len_s = struct.unpack("<Q", b_len_s)[0]
            
            if len_c > 5000 or len_s > 5000:
                print(f"[!] Errore sync al token {i}. Abort.")
                break
                
            bytes_c = self.f_clean_raw.read(len_c)
            bytes_s = self.f_susp_raw.read(len_s)
            
            if bytes_c != bytes_s:
                try: o_str = bytes_c.decode('utf-8', 'ignore')
                except: o_str = repr(bytes_c)
                try: s_str = bytes_s.decode('utf-8', 'ignore')
                except: s_str = repr(bytes_s)
                
                anomalies.append((i, o_str, s_str))

        if not anomalies:
            print("[OK] Vocabolario Semantico Identico.")
        else:
            print(f"[!!!] RILEVATE {len(anomalies)} MANIPOLAZIONI SEMANTICHE (SWAP ATTACK):")
            print(f"{'ID':<6} | {'ORIGINALE':<20} | {'INFETTO'}")
            print("-" * 60)
            for idx, o, s in anomalies[:20]:
                o = o.replace('\n', '\\n')
                s = s.replace('\n', '\\n')
                print(f"{idx:<6} | {o[:20]:<20} -> {s}")
            if len(anomalies) > 20:
                print(f"... e altri {len(anomalies)-20} token.")

    def close(self):
        self.f_clean_raw.close()
        self.f_susp_raw.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ai_model_analysis.py <clean_model.gguf> <suspected_model.gguf>")
    else:
        tool = MasterForensics(sys.argv[1], sys.argv[2])
        try:
            tool.audit_metadata()
            tool.audit_tensors()
            tool.audit_vocabulary_raw()
        finally:
            tool.close()
