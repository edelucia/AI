#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###########################################################################
# Safetensors Forensics v. 1.0                                            #
#                                                                         #
# Author: Emanuele De Lucia                                               #
#                                                                         #
# Feature:                                                                #
# - Singular Value Decomposition (Full & Randomized)                      #
# - Spectral Anomaly Detection (Stable Rank)                              #
# - Benford's Law Divergence Analysis                                     #
# - Polyglot & Entropy Check                                              #
###########################################################################

import os
import sys
import json
import struct
import argparse
import logging
import mmap
import re
import hashlib
import math
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ST-FORENSICS] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib non trovato. La generazione dei grafici sarà disabilitata.")

class SecurityViolation(Exception):
    pass

class SafetensorsForensics:
    MAX_HEADER_SIZE = 100 * 1024 * 1024
    ENTROPY_THRESHOLD_HIGH = 7.95 
    ENTROPY_THRESHOLD_LOW = 0.5
    BENFORD_THRESHOLD = 0.05

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File non trovato: {file_path}")
        
        self.file_size = os.path.getsize(file_path)
        self.header_len = 0
        self.metadata = {}
        self.data_area_start = 0
        self.anomalies = []
        self.file_hash = None
        
        self.plot_data = defaultdict(list) 
        self.spectral_data = defaultdict(list)

    def calculate_file_hash(self):
        logger.info("Calcolo hash SHA-256 del file...")
        sha256_hash = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096 * 1024), b""):
                sha256_hash.update(byte_block)
        self.file_hash = sha256_hash.hexdigest()
        logger.info(f"File Hash: {self.file_hash}")

    def _strict_json_decoder(self, pairs: List[Tuple[Any, Any]]) -> Dict[Any, Any]:
        d = {}
        for k, v in pairs:
            if k in d:
                raise SecurityViolation(f"CRITICO: Chiave duplicata '{k}' nell'header JSON (Parser Attack).")
            d[k] = v
        return d

    def _calculate_entropy(self, data: np.ndarray) -> float:
        byte_data = data.tobytes()
        if len(byte_data) == 0: return 0.0
        counts = np.bincount(np.frombuffer(byte_data, dtype=np.uint8), minlength=256)
        probs = counts[counts > 0] / len(byte_data)
        return -np.sum(probs * np.log2(probs))

    def _calculate_benford_divergence(self, data: np.ndarray) -> float:
        clean_data = data[np.isfinite(data) & (data != 0)]
        if len(clean_data) < 1000:
            return 0.0
            
        abs_data = np.abs(clean_data)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log10_data = np.log10(abs_data)
            first_digits = np.floor(abs_data * (10 ** -np.floor(log10_data))).astype(int)
        
        first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]
        
        if len(first_digits) == 0:
            return 0.0

        counts = np.bincount(first_digits, minlength=10)[1:10]
        observed_freq = counts / np.sum(counts)
        
        digits = np.arange(1, 10)
        expected_freq = np.log10(1 + 1/digits)
        
        sse = np.sum((observed_freq - expected_freq) ** 2)
        return sse

    def _sample_tensor(self, mm: mmap.mmap, start: int, end: int, dtype_input: Any, n_samples: int = 100000) -> np.ndarray:
        dt = np.dtype(dtype_input)
        tensor_len_bytes = end - start
        element_size = dt.itemsize
        total_elements = tensor_len_bytes // element_size

        if total_elements <= n_samples:
            raw_data = mm[self.data_area_start + start : self.data_area_start + end]
            return np.frombuffer(raw_data, dtype=dt)
        
        chunk_size = 4096
        num_chunks = n_samples // chunk_size
        
        samples = []
        for _ in range(num_chunks):
            rand_idx = random.randint(0, total_elements - chunk_size)
            byte_offset = rand_idx * element_size
            
            chunk_start = self.data_area_start + start + byte_offset
            chunk_end = chunk_start + (chunk_size * element_size)
            
            raw_chunk = mm[chunk_start:chunk_end]
            samples.append(np.frombuffer(raw_chunk, dtype=dt))
            
        return np.concatenate(samples)

    def _load_full_tensor_for_svd(self, mm: mmap.mmap, start: int, end: int, dtype_input: Any, shape: List[int]) -> Optional[np.ndarray]:
        dt = np.dtype(dtype_input)
        required_bytes = end - start
        if required_bytes > 2 * 1024 * 1024 * 1024:
            return None

        raw_data = mm[self.data_area_start + start : self.data_area_start + end]
        arr = np.frombuffer(raw_data, dtype=dt)
        try:
            return arr.reshape(shape)
        except ValueError:
            return None

    def _randomized_svd(self, M: np.ndarray, k: int = 10, n_oversamples: int = 10, n_iter: int = 2) -> np.ndarray:
        m, n = M.shape
        target_rank = k + n_oversamples
        
        Omega = np.random.randn(n, target_rank).astype(M.dtype)
        
        Y = M @ Omega
        
        for _ in range(n_iter):
            Y = M @ (M.T @ Y)
            
        Q, _ = np.linalg.qr(Y)
        
        B = Q.T @ M

        s = np.linalg.svd(B, compute_uv=False)
        
        return s

    def validate_structure(self):
        logger.info("Fase 1: Validazione Strutturale...")
        
        with open(self.file_path, 'rb') as f:
            length_bytes = f.read(8)
            if len(length_bytes) != 8:
                raise SecurityViolation("File corrotto/troppo piccolo.")
            
            self.header_len = struct.unpack('<Q', length_bytes)[0]
            if self.header_len > self.MAX_HEADER_SIZE:
                raise SecurityViolation(f"Header Size eccessivo ({self.header_len}).")
            
            self.data_area_start = 8 + self.header_len
            f.seek(8)
            json_bytes = f.read(self.header_len)
            try:
                self.metadata = json.loads(
                    json_bytes.decode('utf-8'), 
                    object_pairs_hook=self._strict_json_decoder
                )
            except json.JSONDecodeError as e:
                raise SecurityViolation(f"JSON Malformato: {str(e)}")

        intervals = []
        for key, info in self.metadata.items():
            if key == "__metadata__": continue
            if "data_offsets" not in info:
                raise SecurityViolation(f"Tensore '{key}' senza offset.")
            intervals.append((info["data_offsets"][0], info["data_offsets"][1]))
        
        intervals.sort(key=lambda x: x[0])
        
        expected_next = 0
        max_end = 0
        for start, end in intervals:
            if start != expected_next:
                raise SecurityViolation(f"Gap dati rilevato a {start}. File non contiguo.")
            expected_next = end
            max_end = max(max_end, end)

        expected_file_size = self.data_area_start + max_end
        if self.file_size != expected_file_size:
            diff = self.file_size - expected_file_size
            if diff > 0:
                raise SecurityViolation(f"CRITICO: {diff} bytes di dati nascosti in coda (Polyglot Vector).")
            else:
                raise SecurityViolation("File troncato.")
        
        logger.info("Struttura: OK.")

    def analyze_semantics_robust(self, z_threshold: float = 5.0):
        logger.info(f"Fase 2: Analisi Semantica, Entropica, Benford e Spettrale (Full/Randomized SVD)...")
        
        pattern_indexer = re.compile(r'\.\d+\.') 
        
        dtype_map = {
            "F32": np.dtype(np.float32), 
            "F16": np.dtype(np.float16), 
            "BF16": np.dtype(np.int16),
            "I64": np.dtype(np.int64),
            "I32": np.dtype(np.int32),
            "I16": np.dtype(np.int16),
            "I8":  np.dtype(np.int8),
            "U8":  np.dtype(np.uint8)
        }
        
        quantized_dtypes = ["I8", "U8", "I32", "I64"]

        with open(self.file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                
                for name, info in self.metadata.items():
                    if name == "__metadata__": continue
                    
                    st_dtype = info['dtype']
                    shape = info['shape']
                    
                    if np.prod(shape) < 1024: continue
                    is_weight_matrix = len(shape) == 2 
                    
                    if st_dtype not in dtype_map: continue

                    is_quantized = (st_dtype in quantized_dtypes) or ("qweight" in name) or ("bnb" in name)
                    
                    try:
                        start, end = info['data_offsets']
                        
                        data_sample = self._sample_tensor(mm, start, end, dtype_map[st_dtype])
                        
                        entropy = self._calculate_entropy(data_sample)
                        
                        if entropy > self.ENTROPY_THRESHOLD_HIGH:
                            if is_quantized:
                                logger.info(f"INFO: High Entropy in Quantized Tensor {name} ({entropy:.4f}). Ignorato (Safe).")
                            else:
                                self.anomalies.append({
                                    "tensor": name,
                                    "type": "High Entropy (Possibile Encrypted Payload)",
                                    "value": entropy,
                                    "z_score": 0.0
                                })
                                logger.critical(f"ALERTA STEGANOGRAFIA: {name} ha entropia {entropy:.4f}")

                        benford_sse = 0.0
                        if not is_quantized and st_dtype in ["F32", "F16"]:
                             data_float_benford = data_sample.astype(np.float64)
                             benford_sse = self._calculate_benford_divergence(data_float_benford)
                             
                             if benford_sse > self.BENFORD_THRESHOLD:
                                 self.anomalies.append({
                                     "tensor": name,
                                     "type": f"Benford Violation (SSE={benford_sse:.4f})",
                                     "value": benford_sse,
                                     "z_score": 0.0
                                 })
                                 logger.warning(f"ANOMALIA BENFORD: {name} devia dalla distribuzione naturale (SSE={benford_sse:.4f})")

                        if not is_weight_matrix: continue

                        data_f64 = data_sample.astype(np.float64)
                        
                        if st_dtype in ["F32", "F16"]:
                            if not np.all(np.isfinite(data_f64)):
                                raise SecurityViolation(f"NaN/Inf rilevato in '{name}'.")

                        median = np.median(data_f64)
                        q25, q75 = np.percentile(data_f64, [25, 75])
                        iqr = q75 - q25
                        
                        group_key = pattern_indexer.sub('.X.', name)
                        
                        self.plot_data[group_key].append({
                            'name': name,
                            'median': median,
                            'iqr': iqr,
                            'entropy': entropy,
                            'benford_sse': benford_sse,
                            'is_quantized': is_quantized
                        })

                        if min(shape) >= 64: 
                            full_tensor = self._load_full_tensor_for_svd(mm, start, end, dtype_map[st_dtype], shape)
                            
                            if full_tensor is not None:
                                use_randomized = min(shape) > 2048
                                self._analyze_spectral_signature(name, full_tensor, group_key, randomized=use_randomized)
                            
                    except Exception as e:
                        logger.warning(f"Errore lettura {name}: {e}")
                        continue

        self._detect_outliers_robust(z_threshold)
        self._detect_spectral_outliers(z_threshold)

    def _analyze_spectral_signature(self, name: str, data: np.ndarray, group_key: str, randomized: bool = False):
        matrix = data.astype(np.float32)

        try:
            if randomized:
                s = self._randomized_svd(matrix, k=10, n_oversamples=10, n_iter=2)
            else:
                s = np.linalg.svd(matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            return

        if len(s) == 0: return

        spectral_norm = s[0]
        frobenius_norm = np.linalg.norm(s)

        if spectral_norm > 0:
            stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2)
        else:
            stable_rank = 0.0

        self.spectral_data[group_key].append({
            'name': name,
            'spectral_norm': spectral_norm,
            'stable_rank': stable_rank,
            'method': 'rSVD' if randomized else 'SVD'
        })

    def _detect_outliers_robust(self, threshold: float):
        for group, items in self.plot_data.items():
            if len(items) < 5: continue 

            group_medians = np.array([x['median'] for x in items])
            base_median = np.median(group_medians)
            iqr_median = np.subtract(*np.percentile(group_medians, [75, 25])) or 1e-9

            group_iqrs = np.array([x['iqr'] for x in items])
            base_iqr = np.median(group_iqrs)
            iqr_iqr = np.subtract(*np.percentile(group_iqrs, [75, 25])) or 1e-9

            scale_factor = 0.7413

            for item in items:
                z_median = abs(item['median'] - base_median) / (iqr_median * scale_factor)
                z_iqr = abs(item['iqr'] - base_iqr) / (iqr_iqr * scale_factor)
                
                item['robust_z'] = max(z_median, z_iqr)

                if item['robust_z'] > threshold:
                    anomaly_msg = []
                    if z_median > threshold: anomaly_msg.append(f"Median Shift (Z={z_median:.1f})")
                    if z_iqr > threshold: anomaly_msg.append(f"Variance Shift (Z={z_iqr:.1f})")
                    
                    self.anomalies.append({
                        "tensor": item['name'],
                        "type": " | ".join(anomaly_msg),
                        "z_score": float(item['robust_z'])
                    })
                    logger.warning(f"ANOMALIA STATISTICA: {item['name']} -> {anomaly_msg}")

    def _detect_spectral_outliers(self, threshold: float):
        for group, items in self.spectral_data.items():
            if len(items) < 5: continue

            ranks = np.array([x['stable_rank'] for x in items])
            med_rank = np.median(ranks)
            iqr_rank = np.subtract(*np.percentile(ranks, [75, 25])) or 1e-9
            
            norms = np.array([x['spectral_norm'] for x in items])
            med_norm = np.median(norms)
            iqr_norm = np.subtract(*np.percentile(norms, [75, 25])) or 1e-9

            scale_factor = 0.7413

            for item in items:
                z_rank = abs(item['stable_rank'] - med_rank) / (iqr_rank * scale_factor)
                z_norm = abs(item['spectral_norm'] - med_norm) / (iqr_norm * scale_factor)
                
                item['spectral_z'] = max(z_rank, z_norm)

                if item['spectral_z'] > threshold:
                    anomaly_msg = []
                    if z_rank > threshold: anomaly_msg.append(f"Stable Rank Anomaly (Z={z_rank:.1f})")
                    if z_norm > threshold: anomaly_msg.append(f"Spectral Norm Spike (Z={z_norm:.1f})")
                    
                    self.anomalies.append({
                        "tensor": item['name'],
                        "type": f"SPECTRAL ({item['method']}): " + " | ".join(anomaly_msg),
                        "z_score": float(item['spectral_z'])
                    })
                    logger.critical(f"ANOMALIA SPETTRALE (Backdoor Profile): {item['name']} -> {anomaly_msg}")

    def generate_forensic_plot(self, output_filename: str):
        if not MATPLOTLIB_AVAILABLE: return
        logger.info(f"Generazione grafico forense: {output_filename}")
        
        groups = sorted([k for k, v in self.spectral_data.items() if len(v) > 10], 
                       key=lambda k: len(self.spectral_data[k]), reverse=True)[:5]
        
        if not groups:
             groups = sorted([k for k, v in self.plot_data.items() if len(v) > 10], 
                       key=lambda k: len(self.plot_data[k]), reverse=True)[:5]
             
        if not groups: return

        plt.figure(figsize=(16, 12))
        plt.style.use('ggplot')
        
        for idx, group in enumerate(groups):
            stat_items = {x['name']: x for x in self.plot_data[group]}
            spec_items = {x['name']: x for x in self.spectral_data[group]}
            
            names = sorted([n for n in stat_items.keys() if n in spec_items], 
                           key=lambda x: int(re.search(r'\.(\d+)\.', x).group(1)) if re.search(r'\.(\d+)\.', x) else x)
            
            if not names: continue

            medians = [stat_items[n]['median'] for n in names]
            stable_ranks = [spec_items[n]['stable_rank'] for n in names]
            
            ax1 = plt.subplot(len(groups), 1, idx + 1)
            
            color = 'tab:blue'
            ax1.set_ylabel('Weight Median', color=color)
            ax1.plot(medians, color=color, alpha=0.5, label='Median (Classic)')
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()  
            color = 'tab:purple'
            ax2.set_ylabel('Stable Rank (SVD)', color=color, fontweight='bold')
            ax2.plot(stable_ranks, color=color, linestyle='-', linewidth=2, label='Stable Rank (SVD)')
            ax2.tick_params(axis='y', labelcolor=color)
            
            anomalies_idx = [i for i, n in enumerate(names) 
                             if stat_items[n].get('robust_z', 0) > 5.0 or spec_items[n].get('spectral_z', 0) > 5.0]
            
            if anomalies_idx:
                ax2.scatter(anomalies_idx, [stable_ranks[i] for i in anomalies_idx], c='red', s=80, zorder=10, marker='x')

            plt.title(f"Spectral & Statistical Analysis: {group}")
            
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        logger.info("Grafico salvato.")

    def run(self, z_threshold: float = 5.0, plot: bool = False) -> int:
        try:
            self.calculate_file_hash()
            self.validate_structure()
            self.analyze_semantics_robust(z_threshold)
            
            if plot:
                self.generate_forensic_plot(self.file_path + ".analysis.png")

            print("\n" + "="*60)
            print("REPORT SAFETENSORS - ENHANCED FORENSICS")
            print("="*60)
            print(f"File: {os.path.basename(self.file_path)}")
            print(f"SHA256: {self.file_hash}")
            
            struct_fail = any('Polyglot' in a.get('type','') for a in self.anomalies)
            print(f"Integrità Strutturale: {'FAIL' if struct_fail else 'PASS'}")
            
            spectral_anomalies = [a for a in self.anomalies if "SPECTRAL" in a['type']]
            benford_anomalies = [a for a in self.anomalies if "Benford" in a['type']]
            critical = [a for a in self.anomalies if "Polyglot" in a['type'] or "Entropy" in a['type']]
            stat_anomalies = [a for a in self.anomalies if a not in spectral_anomalies and a not in critical and a not in benford_anomalies]

            print("-" * 60)
            
            if critical:
                print(f"[!] Potenziali Anomalie CRITICHE ({len(critical)}):")
                for err in critical:
                    print(f" - {err['tensor']}: {err['type']}")
            
            if spectral_anomalies:
                print(f"\n[!] Warning Spettrali ({len(spectral_anomalies)}):")
                for err in spectral_anomalies:
                    print(f" - {err['tensor']}: {err['type']} [Z-Score: {err['z_score']:.1f}]")

            if benford_anomalies:
                print(f"\n[!] Violazioni Legge di Benford ({len(benford_anomalies)}):")
                for err in benford_anomalies:
                    print(f" - {err['tensor']}: {err['type']}")

            if stat_anomalies:
                print(f"\n[!] Warning Statistici ({len(stat_anomalies)}):")
                for warn in stat_anomalies[:5]:
                    print(f" - {warn['tensor']}: {warn['type']}")
                if len(stat_anomalies) > 5: print(f" ... e altri {len(stat_anomalies)-5}.")
            
            if not self.anomalies:
                print("[+] Nessuna anomalia rilevata.")
            
            return 1 if (critical or spectral_anomalies or benford_anomalies) else 0

        except SecurityViolation as e:
            logger.critical(f"POSSIBILE ANOMALIA: {e}")
            return 1
        except Exception as e:
            logger.exception(f"Errore: {e}")
            return 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safetensors Forensics")
    parser.add_argument("file", help="File .safetensors da analizzare")
    parser.add_argument("--threshold", type=float, default=5.0, help="Soglia Robust Z-Score")
    parser.add_argument("--plot", action="store_true", help="Genera grafico (SVD + Stats)")
    args = parser.parse_args()
    forensics = SafetensorsForensics(args.file)
    sys.exit(forensics.run(z_threshold=args.threshold, plot=args.plot))
