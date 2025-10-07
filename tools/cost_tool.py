# cost_tool.py (improved)
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import logging
from functools import lru_cache
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer, util
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from rapidfuzz import fuzz

# Config
SIM_THRESHOLD = 0.45
BASE_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('cost_tool_rag.log', encoding='utf-8')])
logger = logging.getLogger(__name__)

# Gemini / Google client init (optional; tool should work without it)
def init_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY chưa thiết lập. Sử dụng chế độ fallback (LLM sẽ không hoạt động).")
        return None
    model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
    return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0.2, max_output_tokens=512)

_gemini_client = init_gemini_client()

def call_llm(prompt: str) -> str:
    if not _gemini_client:
        logger.debug("LLM không khả dụng, trả empty string từ call_llm.")
        return ""
    try:
        logger.debug("Gọi LLM với prompt: %s", prompt[:300])
        response = _gemini_client.invoke([HumanMessage(content=prompt)])
        text = getattr(response, "content", None) or str(response)
        text = text.strip()
        logger.debug("LLM trả về: %s", text[:400])
        return text
    except Exception as e:
        logger.exception("Lỗi gọi LLM: %s", e)
        return ""

# Load embedding model (cached)
logger.debug("Load embedding model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load DB files
try:
    with open(BASE_DIR / "goi_kham_vip_full.json", "r", encoding="utf-8") as f:
        vip_data = json.load(f)
except FileNotFoundError as e:
    logger.error("Không tìm thấy file dữ liệu trong data/: %s", e)
    raise

packages = vip_data.get("packages", [])

# Build package index
package_index = []
for pkg in packages:
    text_corpus = pkg.get("name","")
    for item in pkg.get("items", []):
        text_corpus += " " + item.get("service_name","")
    package_index.append({
        "id": pkg.get("package_id"),
        "name": pkg.get("name"),
        "price": pkg.get("price", {}),
        "items": pkg.get("items", []),
        "corpus": text_corpus
    })

@lru_cache(maxsize=4096)
def cached_embedding(text: str, to_tensor: bool = False):
    return model.encode(text, convert_to_tensor=to_tensor)

def save_history(session_id: str, entry: dict):
    path = HISTORY_DIR / f"history_{session_id}.json"
    try:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        else:
            existing = []
        existing.append(entry)
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Lưu history cho session %s", session_id)
    except Exception as e:
        logger.exception("Lỗi lưu history: %s", e)

def fuzzy_find_by_query(query: str, top_k: int = 5) -> List[Dict]:
    # loose token overlap fallback
    qtokens = set([t.lower() for t in query.split() if t.strip()])
    loose = []
    for pkg in package_index:
        name_tokens = set([t.lower() for t in pkg["name"].split()])
        overlap = len(qtokens & name_tokens)
        if overlap > 0:
            score = overlap / max(1, len(name_tokens))
            loose.append({
                "id": pkg["id"],
                "name": pkg["name"],
                "price": pkg["price"],
                "items": pkg["items"][:5],
                "relevance_score": round(float(score),4),
                "matched_on": "loose_token"
            })
    loose = sorted(loose, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
    return loose

def search_packages_by_embedding(query: str, top_k: int = 5) -> List[Dict]:
    try:
        corpus = [pkg["corpus"] for pkg in package_index]
        emb_all = model.encode([query] + corpus, convert_to_tensor=True)
        sims = util.cos_sim(emb_all[0], emb_all[1:])[0].cpu().numpy()
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in idxs:
            pkg = package_index[int(idx)]
            results.append({
                "id": pkg["id"],
                "name": pkg["name"],
                "price": pkg["price"],
                "items": pkg["items"][:5],
                "relevance_score": float(sims[int(idx)]),
                "matched_on": "query_fulltext"
            })
        return results
    except Exception as e:
        logger.exception("search_packages_by_embedding lỗi: %s", e)
        return []

def extract_diseases_from_parts(parts: List[str]) -> List[str]:
    # simple heuristic from earlier agent (keep short)
    if not parts:
        return []
    text = "\n".join(parts).lower()
    candidates = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        # detect bullets or numbered lists
        if s.startswith("-") or s.startswith("*") or s[0].isdigit():
            cand = s.lstrip("-*0123456789. ").strip()
            if cand:
                candidates.append(cand)
    # fallback: look for "bệnh", "có thể", "có thể mắc"
    m = []
    for w in ["bệnh", "có thể", "có thể mắc", "có thể là", "các bệnh"]:
        if w in text:
            # split heuristically by commas
            parts_split = text.split(w,1)[1]
            for p in parts_split.replace(";",",").split(","):
                p = p.strip()
                if p:
                    candidates.append(p)
    # dedupe
    cleaned = []
    for c in candidates:
        c2 = c.strip().rstrip(".")
        if c2 and c2 not in cleaned:
            cleaned.append(c2)
    return cleaned

def compute_relevance_simple(pkg_corpus: str, seed_text: str, symptoms: List[str]) -> float:
    try:
        query = f"{seed_text} {' '.join(symptoms)}".strip()
        emb = model.encode([query, pkg_corpus], convert_to_tensor=True)
        score = float(util.cos_sim(emb[0], emb[1]).cpu().numpy()[0][0])
    except Exception:
        score = 0.0
    # Try LLM blend if available (optional)
    if _gemini_client:
        try:
            llm_prompt = f"Trả về điểm từ 0 đến 1 cho mức độ phù hợp của gói mô tả: '{pkg_corpus}' với bệnh '{seed_text}' và triệu chứng {symptoms}."
            resp = call_llm(llm_prompt)
            # try parse leading float
            llm_score = float(resp.strip().split()[0])
            return 0.7*score + 0.3*llm_score
        except Exception:
            pass
    return score

def search_packages_smart(user_query: str, disease_candidates: Optional[List[str]] = None, top_k: int = 5) -> List[Dict]:
    """Multi-strategy: if disease_candidates present, do disease-centric search; else query-based search."""
    try:
        disease_candidates = disease_candidates or []
        results = []
        # Strategy A: disease-centric
        if disease_candidates:
            symptoms = user_query.split()[:20]
            for pkg in package_index:
                best_score = 0.0
                for d in disease_candidates:
                    score = compute_relevance_simple(pkg["corpus"], d, symptoms)
                    if score > best_score:
                        best_score = score
                        best_match = d
                if best_score > 0:
                    results.append({
                        "id": pkg["id"],
                        "name": pkg["name"],
                        "price": pkg["price"],
                        "items": pkg["items"][:5],
                        "relevance_score": round(float(best_score),4),
                        "matched_on": f"disease:{best_match}"
                    })
            results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]
            if results:
                return results
        # Strategy B: embedding query
        emb_results = search_packages_by_embedding(user_query, top_k=top_k)
        if emb_results:
            return emb_results
        # Strategy C: fuzzy token
        return fuzzy_find_by_query(user_query, top_k=top_k)
    except Exception as e:
        logger.exception("search_packages_smart lỗi: %s", e)
        return []

# ==== Tool exposed to agent ====
@tool
async def cost_tool_rag(agent_output: Dict) -> Dict:
    """
    Expected input dict can contain:
      {
        "session_id": str,
        "user_query": str,
        "final_response_parts": list[str],  # optional
        "intent": "cost-only"|"symptom"|"symptom+cost"|"unknown",
        "disease_candidates": list[str]  # optional
      }
    Returns standardized dict:
      {"status": "completed"|"no_match"|"error", "message": "...", "data": {"packages": [...], "input": ...}}
    """
    logger.debug("cost_tool_rag called with keys: %s", list(agent_output.keys()))
    try:
        if not isinstance(agent_output, dict):
            raise ValueError("agent_output phải là dict")

        session_id = agent_output.get("session_id", "unknown")
        user_query = agent_output.get("user_query", "") or ""
        final_response_parts = agent_output.get("final_response_parts", []) or []
        intent = agent_output.get("intent", "unknown")
        disease_candidates = agent_output.get("disease_candidates", []) or []

        # If final_response_parts exists and disease_candidates empty, try to extract diseases
        if final_response_parts and not disease_candidates:
            disease_candidates = extract_diseases_from_parts(final_response_parts)

        # If still empty and intent implies cost-only but query contains explicit disease name,
        # attempt fuzzy extraction from query against package corpuses
        if not disease_candidates and intent in ("cost-only", "unknown"):
            # heuristics: look for keywords like 'viêm', 'ung thư', 'gan', 'tim', 'thần kinh'
            heur = []
            for token in ["viêm", "ung", "gan", "tim", "thần kinh", "dạ dày", "đại trực tràng", "tiêu hóa"]:
                if token in user_query.lower():
                    heur.append(token)
            if heur:
                disease_candidates = [user_query]

        # Search packages smartly
        packages_found = search_packages_smart(user_query, disease_candidates, top_k=5)

        if not packages_found:
            msg = "Không tìm thấy gói khám phù hợp trong dữ liệu."
            result = {"status": "no_match", "message": msg, "data": {"packages": [], "input_query": user_query}}
            save_history(session_id, {"type":"cost_tool_rag", "input": agent_output, "result": result})
            return result

        result = {
            "status": "completed",
            "message": f"Tìm thấy {len(packages_found)} gói khám phù hợp.",
            "data": {
                "input_query": user_query,
                "intent": intent,
                "disease_candidates": disease_candidates,
                "packages": packages_found
            }
        }

        save_history(session_id, {"type":"cost_tool_rag", "input": agent_output, "result": result})
        logger.debug("cost_tool_rag completed: found %d", len(packages_found))
        return result

    except Exception as e:
        logger.exception("Error in cost_tool_rag: %s", e)
        return {"status": "error", "message": str(e), "data": None}
