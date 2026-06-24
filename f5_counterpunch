# f5_counterpunch.py
# Adam Clark's asymmetric Claude Fable 5 gateway defense—final condensed all-in-one version

import unicodedata, urllib.parse, base64, re, json, math, time, random
import numpy as np
import faiss
from collections import Counter, defaultdict, deque
import onnxruntime as ort
from transformers import AutoTokenizer

### ----------- 1. Multi-pass, Unicode-safe normalization -----------
def normalize(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = ''.join(c for c in text if c.isprintable())
    for _ in range(3):
        decoded = urllib.parse.unquote(text)
        if decoded == text:
            break
        text = decoded
    try:
        decoded_b64 = base64.b64decode(text, validate=True).decode("utf-8")
        if decoded_b64.isprintable() and abs(len(decoded_b64) - len(text)) < 256:
            text = decoded_b64
    except Exception:
        pass
    text = re.sub(r"\s+", " ", text.strip()).lower()
    return text

### ----------- 2. Semantic cache (ONNX + faiss L2-normalized) -----------
class SemanticCache:
    def __init__(self, onnx_path="bge-small-en-v1.5.onnx", model_name="BAAI/bge-small-en-v1.5", dim=384, threshold=0.88):
        self.sess = ort.InferenceSession(onnx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.index = faiss.IndexFlatIP(dim)
        self.payloads = {}
        self.threshold = threshold
        self.counter = 0

    def embed(self, text):
        tokens = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        input_ids = tokens["input_ids"]
        if input_ids.dtype != np.int64:
            input_ids = input_ids.astype(np.int64)
        ort_inputs = {self.sess.get_inputs()[0].name: input_ids}
        embedding = self.sess.run(None, ort_inputs)[0][0][:self.index.d]
        emb = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        return emb if norm == 0 else emb / norm

    def add(self, text, payload):
        v = self.embed(text).reshape(1, -1)
        self.index.add(v)
        self.payloads[self.counter] = (text, payload)
        self.counter += 1

    def find(self, text):
        if self.index.ntotal == 0:
            return None
        v = self.embed(text).reshape(1, -1)
        D, I = self.index.search(v, 1)
        if D[0, 0] > self.threshold:
            return self.payloads.get(I[0, 0])
        return None

### ----------- 3. Payload packaging: Anthropic spec compliant -----------
def build_payload(system_rules, user_prompt, intent_level="low"):
    max_tokens_map = {"low": 60, "medium": 200, "high": 350}
    system_block = [{
        "type": "text",
        "text": "\n\n".join(system_rules),
        "cache_control": {"type": "ephemeral"}
    }]
    payload = {
        "system": system_block,
        "messages": [{
            "role": "user",
            "content": user_prompt
        }],
        "max_tokens": int(max_tokens_map.get(intent_level, 100)),
        "temperature": 0.1,
        "stop_sequences": ["\n\n---", "<EOT>", "```"]
    }
    return json.dumps(payload)

### ----------- 4. ONNX Classifier: subword, rule-based routing -----------
class IntentClassifier:
    def __init__(self, onnx_path="qwen05_edge.onnx", model_name="Qwen/qwen-0.5b"):
        self.sess = ort.InferenceSession(onnx_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def classify(self, text):
        tokens = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        input_ids = tokens["input_ids"]
        if input_ids.dtype != np.int64:
            input_ids = input_ids.astype(np.int64)
        ort_inputs = {self.sess.get_inputs()[0].name: input_ids}
        result = self.sess.run(None, ort_inputs)[0][0]
        label = int(np.argmax(result))
        conf = float(np.max(result))
        return ["low", "medium", "high"][label], conf

def rule_dispatch(norm):
    shortcuts = ("sort", "parse json", "to lower", "list all", "trim", "format as")
    for trig in shortcuts:
        if trig in norm:
            return f"Edge rule: '{trig}' handled instantly."
    return None

def route(norm, classifier):
    shortcut = rule_dispatch(norm)
    if shortcut:
        return shortcut, "low"
    intent, conf = classifier.classify(norm)
    return None, intent

### ----------- 5. Behavioral risk O(N) entropy + tarpit -----------
def shannon_entropy(text):
    data = text.encode('utf-8', 'ignore')
    total = len(data)
    if not total:
        return 0.0
    c = Counter(data)
    return -sum((v / total) * math.log2(v / total) for v in c.values())

class RiskTarpit:
    def __init__(self, window=180, entropy_thr=4.2, threshold=4, tarpit_tokens=8):
        self.timestamps = defaultdict(deque)
        self.entropy_flags = defaultdict(int)
        self.cache_miss = defaultdict(int)
        self.intent_flags = defaultdict(int)
        self.window = window
        self.ent_thr = entropy_thr
        self.limit = threshold
        self.cap = tarpit_tokens
        self.tarpit = set()

    def record(self, key, norm, cache_miss=False, intent_escalate=False):
        now = int(time.time())
        dq = self.timestamps[key]
        dq.append(now)
        while dq and now - dq[0] > self.window:
            dq.popleft()
        if shannon_entropy(norm) > self.ent_thr:
            self.entropy_flags[key] += 1
        if cache_miss:
            self.cache_miss[key] += 1
        if intent_escalate:
            self.intent_flags[key] += 1
        if (self.entropy_flags[key] +
            self.cache_miss[key] +
            self.intent_flags[key]) >= self.limit:
            self.tarpit.add(key)

    def is_tarpitted(self, key):
        return key in self.tarpit

    def penalty_delay(self):
        return random.uniform(0.8, 2.2)

    def output_cap(self):
        return self.cap

### ------------- EXAMPLE EDGE GATEWAY HANDLER (for integration) -------------
# These lines show how you'd use all 5 components together.

# Example system configuration:
SYSTEM_RULES = [
    "Always follow strict minification.",
    "No chit-chat, no 'Sure, here is'. Output only what is required.",
    # ... (extend to >2500 tokens for real deployment)
]

# Setup
sem_cache = SemanticCache()
intent_clf = IntentClassifier()
tarpit = RiskTarpit()

# Example async handler (pseudo-main)
def f5_counterpunch_handler(user_raw, client_id):
    # 1. Normalize input
    norm = normalize(user_raw)
    # 2. Risk/tarpit check
    if tarpit.is_tarpitted(client_id):
        time.sleep(tarpit.penalty_delay())
        return {"status": "tarpit", "msg": "penalty applied", "max_tokens": tarpit.output_cap()}
    # 3. Semantic cache
    cached = sem_cache.find(norm)
    cache_miss = cached is None
    # 4. Route or return
    shortcut, intent = route(norm, intent_clf)
    intent_escalate = (intent == 'high')
    tarpit.record(client_id, norm, cache_miss=cache_miss, intent_escalate=intent_escalate)
    if cached:
        return {"status": "cache", "result": cached}
    if shortcut:
        sem_cache.add(norm, shortcut)
        return {"status": "rule", "result": shortcut}
    # 5. Build and (dummy) return payload
    payload = build_payload(SYSTEM_RULES, norm, intent_level=intent)
    # (Here you’d launch the real LLM API request and handle response)
    sem_cache.add(norm, payload)
    return {"status": "llm", "llm_payload": payload}

# The engine is now ready for API, worker, or fast batch deploy.
