"""
validation.py

Post-assembly validation: computes paragraph-level coverage scores and
generates loss reports so that dropped context is detectable and auditable.
"""

from typing import List, Tuple

import numpy as np

from .embeddings import EmbeddingStore
from .utils import split_paragraphs


def compute_coverage(
    original_text: str,
    output_text: str,
    embedding_store: EmbeddingStore,
    threshold: float = 0.85,
) -> dict:
    """
    Compute paragraph-level coverage of the original text in the output.

    For every paragraph in the original, we find the most similar paragraph
    in the output (by cosine similarity of embeddings).  A paragraph is
    "covered" if its best match exceeds *threshold*.

    Returns a dict with:
        coverage_score        – float in [0, 1]
        total_paragraphs      – int
        covered_paragraphs    – int
        uncovered_paragraphs  – list of (paragraph_text, best_similarity)
        paragraph_details     – list of (paragraph_text, best_similarity, matched)
    """
    orig_paragraphs = split_paragraphs(original_text)
    output_paragraphs = split_paragraphs(output_text)

    if not orig_paragraphs:
        return {
            "coverage_score": 1.0,
            "total_paragraphs": 0,
            "covered_paragraphs": 0,
            "uncovered_paragraphs": [],
            "paragraph_details": [],
        }

    # Embed all output paragraphs
    output_embeddings: List[np.ndarray] = []
    for p in output_paragraphs:
        emb = embedding_store.get_embedding(p)
        if emb is not None:
            output_embeddings.append(emb)

    if not output_embeddings:
        return {
            "coverage_score": 0.0,
            "total_paragraphs": len(orig_paragraphs),
            "covered_paragraphs": 0,
            "uncovered_paragraphs": [(p, 0.0) for p in orig_paragraphs],
            "paragraph_details": [(p, 0.0, False) for p in orig_paragraphs],
        }

    output_matrix = np.vstack(output_embeddings).astype(np.float32)

    covered = 0
    uncovered: List[Tuple[str, float]] = []
    details: List[Tuple[str, float, bool]] = []

    for para in orig_paragraphs:
        para_emb = embedding_store.get_embedding(para)
        if para_emb is None:
            uncovered.append((para, 0.0))
            details.append((para, 0.0, False))
            continue

        # Cosine similarity against all output paragraphs
        para_vec = para_emb.reshape(1, -1)
        norms_out = np.linalg.norm(output_matrix, axis=1) + 1e-9
        norm_para = float(np.linalg.norm(para_vec)) + 1e-9
        similarities = (output_matrix @ para_vec.T).flatten() / (norms_out * norm_para)

        best_sim = float(np.max(similarities))
        matched = best_sim >= threshold

        if matched:
            covered += 1
        else:
            uncovered.append((para, best_sim))

        details.append((para, best_sim, matched))

    score = covered / len(orig_paragraphs) if orig_paragraphs else 1.0

    return {
        "coverage_score": score,
        "total_paragraphs": len(orig_paragraphs),
        "covered_paragraphs": covered,
        "uncovered_paragraphs": uncovered,
        "paragraph_details": details,
    }


def generate_loss_report(coverage_result: dict) -> str:
    """
    Generate a human-readable Markdown report of content coverage.
    """
    score = coverage_result["coverage_score"]
    total = coverage_result["total_paragraphs"]
    covered = coverage_result["covered_paragraphs"]
    uncovered = coverage_result["uncovered_paragraphs"]

    lines = [
        "# Coverage Report",
        "",
        f"**Coverage Score:** {score:.1%}",
        f"**Paragraphs Covered:** {covered} / {total}",
        "",
    ]

    if score >= 0.95:
        lines.append("✅ Excellent coverage. Minimal content loss detected.")
    elif score >= 0.85:
        lines.append(
            "⚠️  Good coverage, but some content may have been lost. "
            "Review uncovered paragraphs below."
        )
    else:
        lines.append(
            "❌ Significant content loss detected. "
            "Review and address uncovered paragraphs."
        )

    if uncovered:
        lines.append("")
        lines.append("## Uncovered Paragraphs")
        lines.append("")
        lines.append(
            "The following original paragraphs had no close match "
            "(similarity < threshold) in the output:"
        )
        lines.append("")

        for i, (para, sim) in enumerate(uncovered, 1):
            preview = para[:300] + "…" if len(para) > 300 else para
            lines.append(f"### {i}. (best similarity: {sim:.3f})")
            lines.append("")
            lines.append(f"> {preview}")
            lines.append("")

    return "\n".join(lines)
