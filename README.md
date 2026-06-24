# F5 Counterpunch

**F5 Counterpunch** is an adversarial-proof, edge-native Claude Fable 5 gateway built by Adam Clark (savagetism@icloud.com).  
It fuses Unicode-robust normalization, subword ONNX intent/semantic deduplication, Anthropic-compliant payload handling, and a multi-metric behavioral tarpit to safeguard high-throughput LLM endpoints.

## Features

- NFKC/cascade Unicode normalization and deep decode
- Subword-tokenizer embeddings (bge-small-en-v1.5 ONNX), *not* ASCII or regex
- Near-duplicate semantic detection with in-memory FAISS, <1ms
- Payload builder strictly aligned with Anthropic’s Messages API (no schema errors)
- Instant rule-based shortcut and ONNX intent tiering to crush unnecessary model spend
- Live risk matrix throttling and entropy tarpit for cost and attack defense

## Quickstart

```sh
git clone https://github.com/[your-username]/f5_counterpunch.git
cd f5_counterpunch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txtx
