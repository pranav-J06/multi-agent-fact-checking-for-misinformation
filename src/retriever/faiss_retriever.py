"""
FAISS Evidence Retriever Agent with Temporal-Aware Live Wikipedia Retrieval.
Prioritizes year-specific articles and sentences, preventing past tournaments
(e.g., 2011 World Cup) from contaminating modern queries (e.g., 2023 World Cup).
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
import json
import urllib.request
import urllib.parse
import re
import hashlib
import logging
from rank_bm25 import BM25Okapi

from src.config import RETRIEVAL_TOP_K, EMBEDDING_DIM, FAISS_INDEX_PATH
from src.retriever.embedder import DenseEmbedder
from data.data_loader import load_wiki_corpus

logger = logging.getLogger(__name__)

USER_AGENT = "VeritasAI-FactChecker/1.0 (Educational NLP Research Project; contact: pranav.jahagirdar2024@vitstudent.ac.in)"

class FAISSEvidenceRetriever:
    def __init__(self, top_k: int = RETRIEVAL_TOP_K, index_path: Path = FAISS_INDEX_PATH, enable_fallback: bool = True):
        self.top_k = top_k
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(".meta.json")
        self.embedder = DenseEmbedder()
        self.corpus: List[Dict[str, Any]] = []
        self.index: faiss.IndexFlatIP = None
        self.local_bm25: BM25Okapi = None
        self._sentence_embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_file = self.index_path.parent / "wiki_article_cache.json"
        self._article_cache: Dict[str, List[Dict[str, Any]]] = self._load_article_cache()
        self.enable_fallback = enable_fallback
        self.gemini_fallback = None
        if self.enable_fallback:
            try:
                from src.retriever.gemini_fallback import GeminiFallbackRetriever
                self.gemini_fallback = GeminiFallbackRetriever()
            except Exception as e:
                logger.debug(f"[Retriever] Gemini fallback not initialized: {e}")
        self._load_or_build_index()

    def _load_article_cache(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_article_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._article_cache, f)
        except Exception:
            pass

    def _get_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._sentence_embedding_cache]
        if missing:
            new_embs = self.embedder.embed_texts(missing)
            for t, emb in zip(missing, new_embs):
                self._sentence_embedding_cache[t] = emb
        return np.array([self._sentence_embedding_cache[t] for t in texts])

    def _compute_corpus_hash(self) -> str:
        """Computes a SHA-256 fingerprint of the loaded corpus content."""
        content_str = json.dumps(self.corpus, sort_keys=True)
        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    def _load_or_build_index(self):
        """Loads existing FAISS index if valid and matching corpus hash, else rebuilds."""
        self.corpus = load_wiki_corpus()
        corpus_hash = self._compute_corpus_hash()

        # Build local Okapi BM25 index over the corpus sentences for hybrid search
        corpus_tokens = [re.findall(r'\b\w+\b', doc["text"].lower()) for doc in self.corpus]
        self.local_bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

        if self.index_path.exists() and self.meta_path.exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("corpus_hash") == corpus_hash and meta.get("ntotal") == len(self.corpus):
                    loaded_index = faiss.read_index(str(self.index_path))
                    if loaded_index.ntotal == len(self.corpus):
                        self.index = loaded_index
                        return
            except Exception as e:
                print(f"[Retriever] Cache verification failed ({e}), rebuilding index...")

        self._build_index(corpus_hash)

    def _build_index(self, corpus_hash: str = None):
        """Encodes all corpus sentences and adds them to a FAISS Flat IP index."""
        if corpus_hash is None:
            corpus_hash = self._compute_corpus_hash()
        texts = [doc["text"] for doc in self.corpus]
        embeddings = self.embedder.embed_texts(texts)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"corpus_hash": corpus_hash, "ntotal": self.index.ntotal}, f)
        print(f"[Retriever] Built and cached FAISS index with {self.index.ntotal} documents at {self.index_path}")

    def _extract_search_terms(self, query: str) -> str:
        """Extracts key named entities, nouns, and temporal anchors to improve Wikipedia search precision."""
        clean = query.strip().rstrip(".!?")
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', clean)
        target_year = years[0] if years else ""

        stopwords = {
            "a", "an", "the", "and", "or", "but", "while", "as", "well", "is", "was", "were",
            "are", "been", "be", "have", "has", "had", "did", "do", "does", "not", "no",
            "never", "dint", "didn't", "that", "this", "these", "those", "of", "in", "on",
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "finished", "occurred", "stated", "sub-claim", "claim", "events", "described"
        }

        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b', clean)
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', clean)
        informative_words = [w for w in words if w.lower() not in stopwords]

        components = []
        if proper_nouns:
            components.extend(proper_nouns[:3])
        if target_year and target_year not in " ".join(components):
            components.append(target_year)
        for w in informative_words:
            if w.lower() not in " ".join(components).lower():
                components.append(w)
            if len(components) >= 5:
                break

        search_q = " ".join(components).strip()
        return search_q if len(search_q) >= 3 else clean

    def _fetch_article_sentences(self, query: str) -> List[Dict[str, Any]]:
        """Fetches lead sentences for a topic from Wikipedia with temporal prioritization."""
        clean_query = query.strip().rstrip(".!?")
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

        # Extract explicit 4-digit year from query (e.g. 2023)
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', clean_query)
        target_year = years[0] if years else None

        cache_key = f"{clean_query}_{target_year}"
        if cache_key in self._article_cache:
            return self._article_cache[cache_key]

        try:
            search_query = self._extract_search_terms(clean_query)
            search_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={urllib.parse.quote(search_query)}&format=json"
            )
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return []

            # Deprioritize satellite/qualifier/records articles unless explicitly queried
            satellite_pattern = r'\b(Qualifier|Qualification|League \d+|Challenge League|Records|List of|Timeline|Draft|Statistics)\b'
            query_wants_satellite = bool(re.search(satellite_pattern, clean_query, re.IGNORECASE))

            def is_main_article(t: str) -> bool:
                if query_wants_satellite:
                    return True
                return not bool(re.search(satellite_pattern, t, re.IGNORECASE))

            main_results = [r for r in search_results if is_main_article(r["title"])]
            fallback_results = [r for r in search_results if not is_main_article(r["title"])]
            ordered_search = main_results + fallback_results

            # Prioritize titles containing target_year (e.g. '2023 Cricket World Cup')
            if target_year:
                exact_year_titles = [r["title"] for r in ordered_search if target_year in r["title"]]
                other_titles = [r["title"] for r in ordered_search if target_year not in r["title"]]
                candidate_titles = (exact_year_titles[:2] + other_titles[:1])[:2]
            else:
                candidate_titles = [r["title"] for r in ordered_search[:2]]

            candidate_sentences = []
            for title in candidate_titles:
                extract_url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                    f"&exintro=1&explaintext=1&titles={urllib.parse.quote(title)}&format=json"
                )
                req_ext = urllib.request.Request(extract_url, headers=headers)
                try:
                    with urllib.request.urlopen(req_ext, timeout=3.5) as resp_ext:
                        data_ext = json.loads(resp_ext.read().decode("utf-8"))
                    pages = data_ext.get("query", {}).get("pages", {})
                    for page in pages.values():
                        raw_text = page.get("extract", "")
                        if not raw_text:
                            continue
                        parts = re.split(r'(?<=[.!?])\s+', raw_text)
                        for p in parts:
                            cleaned = p.strip()
                            if len(cleaned.split()) >= 4:
                                # Temporal Filter: If query has target_year, discard sentences discussing different years
                                other_years = [y for y in re.findall(r'\b(19\d{2}|20\d{2})\b', cleaned) if y != target_year]
                                if target_year and other_years and (target_year not in cleaned and target_year not in title):
                                    continue
                                candidate_sentences.append({
                                    "title": title,
                                    "text": cleaned,
                                    "id": f"wiki_{title}_{len(candidate_sentences)+1}"
                                })
                except Exception:
                    continue

            self._article_cache[cache_key] = candidate_sentences
            self._save_article_cache()
            return candidate_sentences

        except Exception as e:
            logger.warning(f"[Retriever] Live Wikipedia lookup failed: {e}")
            return []

    def retrieve(self, sub_claims: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k evidence sentences for each sub-claim with year-aware ranking.
        """
        k = top_k or self.top_k
        if not sub_claims:
            return []

        sub_embeddings = self.embedder.embed_texts(sub_claims)
        distances, indices = self.index.search(sub_embeddings, k)

        results = []

        for i, sub_claim in enumerate(sub_claims):
            retrieved_items = []
            scores = distances[i]
            idx_list = indices[i]

            sub_tokens = re.findall(r'\b\w+\b', sub_claim.lower())
            local_bm25_raw = self.local_bm25.get_scores(sub_tokens) if self.local_bm25 else None
            local_bm25_max = np.max(local_bm25_raw) if local_bm25_raw is not None and np.max(local_bm25_raw) > 0 else 1.0

            for score, doc_idx in zip(scores, idx_list):
                if doc_idx < 0 or doc_idx >= len(self.corpus):
                    continue
                doc = self.corpus[doc_idx]
                dense_s = float(np.clip(score, 0.0, 1.0))
                sparse_s = float(np.clip(local_bm25_raw[doc_idx] / local_bm25_max, 0.0, 1.0)) if local_bm25_raw is not None else dense_s
                combined_s = 0.75 * dense_s + 0.25 * sparse_s
                retrieved_items.append({
                    "id": doc.get("id", f"doc_{doc_idx}"),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "score": round(combined_s, 4),
                    "dense_score": round(dense_s, 4),
                    "bm25_score": round(sparse_s, 4)
                })

            retrieved_items.sort(key=lambda x: x["score"], reverse=True)
            top_local_score = float(retrieved_items[0]["score"]) if retrieved_items else 0.0

            # Sub-Claim Isolated Live Wikipedia Retrieval
            if top_local_score < 0.55:
                live_candidates = self._fetch_article_sentences(sub_claim)
                if live_candidates:
                    sub_emb = sub_embeddings[i:i+1]
                    texts = [c["text"] for c in live_candidates]
                    sent_embs = self._get_sentence_embeddings(texts)
                    sims = (sub_emb @ sent_embs.T)[0]

                    # Okapi BM25 scoring across live candidate pool
                    cand_tokens = [re.findall(r'\b\w+\b', c["text"].lower()) for c in live_candidates]
                    bm25_live = BM25Okapi(cand_tokens) if cand_tokens else None
                    bm25_raw = bm25_live.get_scores(sub_tokens) if bm25_live else np.zeros(len(live_candidates))
                    max_b = np.max(bm25_raw) if len(bm25_raw) > 0 and np.max(bm25_raw) > 0 else 1.0
                    bm25_norm = np.clip(bm25_raw / max_b, 0.0, 1.0)

                    query_words = set(re.findall(r'\b\w{3,}\b', sub_claim.lower())) - {
                        "the", "and", "for", "with", "that", "this", "from", "was", "were", "are", "been"
                    }
                    years = re.findall(r'\b(19\d{2}|20\d{2})\b', sub_claim)
                    target_year = years[0] if years else None

                    ranked_scores = []
                    for s_idx, sim in enumerate(sims):
                        cand = live_candidates[s_idx]
                        txt = cand["text"].lower()
                        title_txt = cand.get("title", "").lower()
                        # Hybrid ranking: 70% dense cosine similarity + 30% Okapi BM25 score
                        score_mod = 0.70 * float(sim) + 0.30 * float(bm25_norm[s_idx])

                        # 1. Main Title Alignment Bonus
                        title_words = set(re.findall(r'\b\w{3,}\b', title_txt))
                        if query_words and title_words:
                            t_overlap = len(query_words & title_words) / len(query_words)
                            score_mod += 0.08 * t_overlap

                        # 2. Satellite Title Deprioritization
                        if re.search(r'\b(qualifier|qualification|league \d+|challenge|records|list of)\b', title_txt):
                            score_mod -= 0.15

                        # 3. Temporal alignment
                        if target_year:
                            if target_year in txt or target_year in title_txt:
                                score_mod += 0.05
                            else:
                                other_years = [y for y in re.findall(r'\b(19\d{2}|20\d{2})\b', txt) if y != target_year]
                                if other_years:
                                    score_mod -= 0.08

                        ranked_scores.append(score_mod)

                    top_ranked = np.argsort(ranked_scores)[::-1][:k]

                    live_items = []
                    for r_idx in top_ranked:
                        raw_s = float(np.clip(sims[r_idx], 0.0, 1.0))
                        live_items.append({
                            "id": live_candidates[r_idx]["id"],
                            "title": live_candidates[r_idx]["title"],
                            "text": live_candidates[r_idx]["text"],
                            "score": raw_s
                        })

                    if live_items and live_items[0]["score"] > top_local_score:
                        retrieved_items = live_items
                        top_local_score = live_items[0]["score"]

            top_score = top_local_score

            # Seamless Tier-2 Fallback: If Wikipedia search yielded inconclusive evidence (< 0.50 similarity)
            if self.enable_fallback and top_score < 0.50 and self.gemini_fallback and self.gemini_fallback.is_available():
                fallback_items = self.gemini_fallback.retrieve_reference_sentences(sub_claim)
                if fallback_items:
                    fb_texts = [item["text"] for item in fallback_items]
                    fb_embs = self.embedder.embed_texts(fb_texts)
                    sub_emb = sub_embeddings[i:i+1]
                    fb_sims = (sub_emb @ fb_embs.T)[0]
                    for f_idx, item in enumerate(fallback_items):
                        item["score"] = float(np.clip(fb_sims[f_idx], 0.0, 1.0))
                    top_score = float(max(fb_sims)) if len(fb_sims) > 0 else 0.65
                    retrieved_items = fallback_items

            concat_text = "\n".join([f"- [{item['title']}] {item['text']}" for item in retrieved_items[:3]])

            results.append({
                "sub_claim": sub_claim,
                "top_evidence": retrieved_items,
                "top_score": top_score,
                "concatenated_evidence": concat_text
            })

        return results
