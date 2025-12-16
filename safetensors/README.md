- Overview

While the .safetensors format solved the Remote Code Execution (RCE) vulnerability inherent in Pickle files, it is not immune to other vectors of attack. Safetensors Forensics is a Python utility designed to analyze the integrity and mathematical consistency of model files.

It detects potential threats such as:

   * Polyglot Files: Hidden data appended to the file (e.g., zip archives or executables hiding in valid model files).

   * Steganography: High-entropy payloads concealed within tensor data.

   * Model Poisoning/Backdoors: Spectral anomalies (via SVD) that suggest tampering with weight matrices.

   * Parser Exploits: Malformed JSON headers designed to trick loaders (Duplicate keys, JSON smuggling).

- Key Features

   * Structure & Polyglot Check: Validates header size, strict JSON parsing, and detects "trailing garbage" or data gaps often used for hiding payloads.

   * Spectral Anomaly Detection: Uses Singular Value Decomposition (SVD) and Randomized SVD to calculate the Stable Rank of weight matrices, identifying layers that deviate mathematically from their neighbors.

   * Entropy Analysis: Scans tensors for high-entropy blocks that indicate encrypted payloads rather than natural neural weights.

   * Benford's Law Divergence: Checks if float weights follow the natural distribution expected in trained models.

   * Robust Outlier Detection: Uses robust Z-Scores (Median/IQR) to flag statistical anomalies while ignoring minor noise.

   * Visual Forensics: Generates plots comparing Weight Medians vs. Stable Ranks to visually spot outliers (requires matplotlib).

- Installation
  
   * Prerequisites

       Python 3.8+

       RAM: Sufficient to load the largest single tensor in the model (nb the whole file is not loaded at once).

   * Dependencies

       Install the required packages:

       pip install numpy matplotlib

    * Usage

       Run the script directly on a .safetensors file.

    * Basic Analysis

       python safetensors_analysis.py /path/to/model.safetensors

    * Advanced Options

       Generate a forensic plot and adjust the sensitivity threshold:

       python safetensors_analysis.py model.safetensors --plot --threshold 4.0

        --plot: Saves a .png chart showing the spectral and statistical profile of the model layers.

        --threshold: Sets the Robust Z-Score limit for flagging anomalies (Default: 5.0).

    * Interpretation of Results

        The tool outputs a report categorized by severity:

        CRITICAL (Security Violation): Structural issues like trailing bytes (Polyglot) or header manipulation. Do not load these files.

        SPECTRAL WARNINGS: Anomalies in the Stable Rank. Could indicate a backdoor or a poorly trained layer.

        ENTROPY WARNINGS: Tensors that look like random noise or encrypted data (potential steganography).

        BENFORD/STATISTICAL WARNINGS: Deviations from expected distribution. Common in fine-tuned models but worth noting.

- Limitations & Known Issues

This tool performs heuristic analysis. It requires human interpretation.

    * Quantization Blindness: The script may skip or have reduced sensitivity for quantized types (Int8, Int4, qweight) as their statistical properties differ significantly from Float16/32.

    * False Positives (Architecture): Some valid architectural choices (like Input Embeddings or specific output heads) naturally have weird spectral signatures. To really dial down structural false positives, we would need a 'whitelist' of spectral signatures for established architectures. If the script could recognize that a specific skew in Layer 0 is just a standard quirk of Llama-3, and not a tamper attempt, we’d see far fewer false alarms. To be implemented in V2.

    * Benford's Law: Heavily fine-tuned models (RLHF/DPO) or "uncensored" models often naturally violate Benford's Law. These warnings should be treated with caution.

- Roadmap

    1) Implementation of Architectural Whitelisting to reduce false positives on Embeddings/Heads.

    2) Better support for GGUF/Quantized formats.

    3) Batch scanning mode for directory analysis.

- License & Credits

    * Author: Emanuele De Lucia License: MIT

    * Disclaimer: This tool is for educational and forensic purposes. It does not guarantee that a model is safe to use.
