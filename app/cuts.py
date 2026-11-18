"""Exact acoustic cut-point scoring for an already decoded caption segment.

The audio editor knows how the caption should be split between two words and
wants the frame boundary that best matches that split.  This module answers the
question with the *exact* CTC path mass, not with the best single path and not
with a beam approximation:

For a segment of frames ``start..end`` (inclusive), a caption ``text`` and a
split after ``split`` characters, define

* ``Z`` = total probability of **all** label paths over the whole segment whose
  CTC collapse is exactly ``text``;
* ``N(b)`` = total probability of all paths over the segment with

  - frame ``b - 1`` blank (the frame immediately before the cut),
  - frames ``start..b-1`` collapsing to ``text[:split]`` (the first half),
  - frames ``b..end`` collapsing to ``text[split:]`` (the second half).

Because the frame before the cut is blank, ``N(b)`` is a subset of the paths
counted by ``Z``: the two halves cannot merge across the boundary.  The score of
boundary ``b`` is the conditional probability ``N(b) / Z``.  Empty sides are
excluded, so ``start < b <= end``.

Both ``Z`` and the two factors of ``N(b)`` are computed with the CTC forward
graph in :mod:`app.decoder`.  That graph enforces the CTC rule that a repeated
character needs an intervening blank, and it propagates ``log 0 = -inf``
exactly, so impossible paths contribute nothing (they are not approximated).

Each frame is normalised to sum to one before scoring.  A frame whose entries
are all zero cannot be normalised; that is reported instead of silently
inventing a distribution.  If no path collapses to ``text`` at all, ``Z`` is
zero and no boundary score exists.

The returned probability is therefore of the form
``P(cut at b | this segment decodes to text)``; it is not rescaled by the tiny
absolute probability of the caption itself.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .decoder import (
    DecodeError,
    compute_prefix_alpha,
    prefix_alpha_scores,
)

BLANK_INDEX = 0
CUT_THRESHOLD = 0.8
# Two mathematically equal boundary probabilities can differ in the last few
# bits because the two halves take different log-sum-exp paths.  Such values
# count as a tie, and the earliest boundary wins.
TIE_REL_TOL = 1e-12
TIE_ABS_TOL = 1e-15
MIN_SEGMENT_FRAMES = 2
MAX_SEGMENT_FRAMES = 64
MIN_TEXT_CHARS = 2
MAX_TEXT_CHARS = 16


def normalise_frame_probs(probs: np.ndarray) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Normalise each frame to sum to one.

    Returns ``(normalised, None)`` on success or ``(None, t)`` when the frame at
    local index ``t`` is all zeros and therefore has no distribution.  A frame
    with a non-zero total is divided by that total; individual zeros stay zero
    and become ``-inf`` in log space later.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise DecodeError("片段概率必须是二维数组")
    totals = probs.sum(axis=1)
    if np.any(totals <= 0.0):
        zero_index = int(np.flatnonzero(totals <= 0.0)[0])
        return None, zero_index
    normalised = probs / totals[:, None]
    return normalised, None


def _log_probs(normalised: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(normalised)


def _finite(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def validate_cut_inputs(
    vocab: Sequence[str],
    start_frame: int,
    end_frame: int,
    text: str,
    split: int,
) -> list[int]:
    """Validate caption / split arguments and return the caption's label ids.

    Only shape-independent rules live here; the caller checks that the segment
    is inside the frames that have actually been advanced.
    """
    if isinstance(start_frame, bool) or not isinstance(start_frame, int):
        raise DecodeError("起始帧必须是整数")
    if isinstance(end_frame, bool) or not isinstance(end_frame, int):
        raise DecodeError("结束帧必须是整数")
    if start_frame < 0:
        raise DecodeError("起始帧不能为负数")
    if end_frame < start_frame:
        raise DecodeError("结束帧不能早于起始帧")
    length = end_frame - start_frame + 1
    if length < MIN_SEGMENT_FRAMES or length > MAX_SEGMENT_FRAMES:
        raise DecodeError(
            f"片段必须包含 {MIN_SEGMENT_FRAMES}～{MAX_SEGMENT_FRAMES} 帧，当前为 {length} 帧"
        )

    if not isinstance(text, str):
        raise DecodeError("字幕必须是字符串")
    if len(text) < MIN_TEXT_CHARS or len(text) > MAX_TEXT_CHARS:
        raise DecodeError(
            f"字幕必须包含 {MIN_TEXT_CHARS}～{MAX_TEXT_CHARS} 个字符，当前为 {len(text)} 个"
        )
    token_index = {token: index for index, token in enumerate(vocab)}
    ids: list[int] = []
    for position, char in enumerate(text):
        index = token_index.get(char)
        if index is None or index == BLANK_INDEX:
            raise DecodeError(f"字幕第 {position + 1} 个字符“{char}”不在词表内")
        ids.append(index)

    if isinstance(split, bool) or not isinstance(split, int):
        raise DecodeError("切分处必须是整数")
    if split < 1 or split >= len(text):
        raise DecodeError(f"切分处必须在 1～{len(text) - 1} 之间")
    return ids


def compute_cut_suggestion(
    segment_probs: np.ndarray,
    vocab: Sequence[str],
    start_frame: int,
    end_frame: int,
    text: str,
    split: int,
) -> dict:
    """Score every legal boundary of one already advanced segment.

    ``segment_probs`` is the raw ``(end - start + 1, len(vocab))`` probability
    block.  ``start_frame`` / ``end_frame`` are absolute inclusive frame
    indices; they are echoed into the result so a client can prove which inputs
    a suggestion belongs to.
    """
    ids = validate_cut_inputs(vocab, start_frame, end_frame, text, split)
    probs = np.asarray(segment_probs, dtype=np.float64)
    length = end_frame - start_frame + 1
    if probs.shape != (length, len(vocab)):
        raise DecodeError("片段帧数与词表大小不匹配")

    normalised, zero_index = normalise_frame_probs(probs)
    base = {
        "status": "ok",
        "threshold": CUT_THRESHOLD,
        "inputs": {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": length,
            "text": text,
            "split": split,
            "first_half": text[:split],
            "second_half": text[split:],
        },
        "best": None,
        "boundaries": [],
        "normalised": zero_index is None,
        "suitable": False,
        "message": "",
    }

    if normalised is None:
        assert zero_index is not None
        result = dict(base)
        result["status"] = "all_zero"
        result["zero_frame"] = start_frame + zero_index
        result["message"] = (
            f"片段内第 {start_frame + zero_index} 帧所有概率均为 0，"
            "无法逐帧归一化，因此不能计算切点。"
        )
        return result

    logp = _log_probs(normalised)
    total_logp, _, _ = prefix_alpha_scores(compute_prefix_alpha(logp, ids))

    if not math.isfinite(total_logp):
        result = dict(base)
        result["status"] = "unreachable"
        result["full_text_logp"] = None
        result["message"] = (
            f"逐帧归一化后，该片段内没有任何 CTC 路径能折叠为全文“{text}”；"
            "可能所需字符在整段中概率为 0，或重复同字之间缺少可用的 blank。"
        )
        return result

    base["full_text_logp"] = float(total_logp)

    boundaries: list[dict] = []
    best: Optional[dict] = None
    best_prob = -1.0

    def is_tie(candidate: float, current: float) -> bool:
        """Whether two boundary probabilities are equal up to rounding.

        Exact mathematical ties are common in symmetric test patterns, and the
        two halves reach the same value through different log-sum-exp paths.
        Treating near-equal values as a tie makes "earliest boundary wins"
        meaningful instead of letting the last-bit noise decide it.
        """
        return abs(candidate - current) <= max(
            TIE_ABS_TOL, TIE_REL_TOL * max(abs(candidate), abs(current))
        )

    # b is the first frame of the right half, so the frame before the cut is
    # b - 1 and both halves contain at least one frame.
    for frame in range(start_frame + 1, end_frame + 1):
        left = frame - start_frame
        left_alpha = compute_prefix_alpha(logp[:left], ids[:split])
        _, left_blank_logp, _ = prefix_alpha_scores(left_alpha)
        right_alpha = compute_prefix_alpha(logp[left:], ids[split:])
        right_total_logp, _, _ = prefix_alpha_scores(right_alpha)

        if math.isfinite(left_blank_logp) and math.isfinite(right_total_logp):
            joint_logp = float(left_blank_logp + right_total_logp)
            conditional_logp = joint_logp - float(total_logp)
            # The numerator is a subset of the denominator; allow only tiny
            # floating-point overshoot above 1.
            probability = min(1.0, math.exp(conditional_logp))
            if probability < 0.0:  # pragma: no cover - exp is non-negative
                probability = 0.0
        else:
            joint_logp = None
            conditional_logp = None
            probability = 0.0

        entry = {
            "cut_frame": frame,
            "prev_frame": frame - 1,
            "probability": probability,
            "log_probability": _finite(conditional_logp) if conditional_logp is not None else None,
            "left_blank_logp": _finite(left_blank_logp),
            "right_logp": _finite(right_total_logp),
        }
        boundaries.append(entry)

        # Strict improvement only: an exact (or rounding-level) tie keeps the
        # boundary found earlier in the scan, i.e. the earliest frame.
        if probability > best_prob and not is_tie(probability, best_prob):
            best_prob = probability
            best = entry
        elif best is None:
            best_prob = probability
            best = entry

    base["boundaries"] = boundaries
    base["best"] = best

    if best is None:
        base["message"] = "片段太短，没有合法边界（左右两侧都必须至少有一帧）。"
        return base
    if best_prob <= 0.0:
        base["message"] = (
            f"全文“{text}”可达，但没有任何边界同时满足"
            "“左侧为前半句、边界前帧为 blank、右侧为后半句”，"
            f"所有边界条件概率均为 0（阈值 {CUT_THRESHOLD:.1f}），不宜切开。"
        )
        return base

    # ``best_frame`` is the first frame of the right half, i.e. the acoustic
    # cut point lies immediately before it.
    base["suitable"] = best_prob >= CUT_THRESHOLD
    if base["suitable"]:
        base["message"] = (
            f"建议在第 {best['cut_frame']} 帧前切开"
            f"（左侧 {start_frame}～{best['prev_frame']} 帧，右侧 "
            f"{best['cut_frame']}～{end_frame} 帧）：条件概率 "
            f"{best_prob:.4f} ≥ {CUT_THRESHOLD:.1f}。"
        )
    else:
        base["message"] = (
            f"最高条件概率为 {best_prob:.4f}，低于阈值 {CUT_THRESHOLD:.1f}，"
            "不宜按当前片段 / 字幕 / 切分处切开；请调整后再试。"
        )
    return base
