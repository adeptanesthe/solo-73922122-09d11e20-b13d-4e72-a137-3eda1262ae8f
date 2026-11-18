"""Exact cut-point scoring: edge cases and brute-force agreement.

The scorer is the only place where an approximate CTC treatment would be easy
to slip in (best path, beam, greedy segmentation).  These tests therefore
compare the conditional boundary probabilities with full path enumeration on
tiny inputs, and pin down the documented edge cases.
"""

import itertools
import math

import numpy as np
import pytest

from app.cuts import (
    CUT_THRESHOLD,
    compute_cut_suggestion,
    normalise_frame_probs,
    validate_cut_inputs,
)
from app.decoder import DecodeError

VOCAB = ["<blank>", "a", "b"]


def collapse(path, vocab=VOCAB):
    out = []
    previous = None
    for index in path:
        if index != 0 and index != previous:
            out.append(vocab[index])
        previous = index
    return "".join(out)


def brute_force_boundaries(frames, text, split, vocab=VOCAB):
    """Enumerate every path; return (Z, {cut_frame: numerator})."""
    probs = np.asarray(frames, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    total = int(probs.shape[0])
    z = 0.0
    numerators = {frame: 0.0 for frame in range(1, total)}
    for path in itertools.product(range(len(vocab)), repeat=total):
        probability = 1.0
        for t, k in enumerate(path):
            probability *= probs[t][k]
        if probability == 0.0:
            continue
        if collapse(path, vocab) == text:
            z += probability
        for frame in range(1, total):
            if path[frame - 1] != 0:
                continue
            if collapse(path[:frame], vocab) != text[:split]:
                continue
            if collapse(path[frame:], vocab) != text[split:]:
                continue
            numerators[frame] += probability
    return z, numerators


def suggest(frames, start, end, text, split, vocab=VOCAB):
    return compute_cut_suggestion(
        np.asarray(frames, dtype=np.float64), vocab, start, end, text, split
    )


# ------------------------------------------------------------------ correctness


def test_conditional_probabilities_match_path_enumeration():
    frames = [
        [0.6, 0.4, 0.0],
        [0.1, 0.8, 0.1],
        [0.5, 0.0, 0.5],
        [0.2, 0.3, 0.5],
        [0.7, 0.2, 0.1],
    ]
    z, numerators = brute_force_boundaries(frames, "ab", 1)
    assert z > 0
    result = suggest(frames, 0, 4, "ab", 1)
    assert result["status"] == "ok"
    assert result["full_text_logp"] == pytest.approx(math.log(z), abs=1e-12)
    assert len(result["boundaries"]) == 4
    for entry in result["boundaries"]:
        expected = numerators[entry["cut_frame"]] / z
        assert entry["probability"] == pytest.approx(expected, abs=1e-12)
        if expected == 0.0:
            assert entry["log_probability"] is None
        else:
            assert entry["log_probability"] == pytest.approx(math.log(expected), abs=1e-12)


def test_absolute_frame_offsets_do_not_change_scores():
    frames = [
        [0.6, 0.4, 0.0],
        [0.1, 0.8, 0.1],
        [0.5, 0.0, 0.5],
        [0.2, 0.3, 0.5],
    ]
    prefix = [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]]
    whole = prefix + frames
    local = suggest(frames, 0, 3, "ab", 1)
    shifted = suggest(whole[2:], 2, 5, "ab", 1)
    assert shifted["inputs"]["start_frame"] == 2
    assert shifted["inputs"]["end_frame"] == 5
    assert [b["cut_frame"] for b in shifted["boundaries"]] == [3, 4, 5]
    for local_entry, shifted_entry in zip(local["boundaries"], shifted["boundaries"]):
        assert shifted_entry["cut_frame"] == local_entry["cut_frame"] + 2
        assert shifted_entry["probability"] == pytest.approx(
            local_entry["probability"], abs=1e-12
        )


def test_frame_normalisation_is_applied_before_scoring():
    """Rows are not distributions on input; each row is divided by its sum."""
    frames = [
        [6.0, 4.0, 0.0],
        [1.0, 8.0, 1.0],
        [5.0, 0.0, 5.0],
    ]
    normalised = [[0.6, 0.4, 0.0], [0.1, 0.8, 0.1], [0.5, 0.0, 0.5]]
    scaled = suggest(frames, 0, 2, "ab", 1)
    reference = suggest(normalised, 0, 2, "ab", 1)
    assert scaled["status"] == reference["status"] == "ok"
    for a, b in zip(scaled["boundaries"], reference["boundaries"]):
        assert a["probability"] == pytest.approx(b["probability"], abs=1e-12)


def test_recommends_highest_probability_and_reports_it():
    # "a" is clear at frame 0, frame 1 is blank-dominant, then "b"; the only
    # boundary whose preceding frame is blank is before frame 2.
    frames = [
        [0.005, 0.99, 0.005],
        [0.98, 0.01, 0.01],
        [0.005, 0.005, 0.99],
        [0.99, 0.005, 0.005],
        [0.99, 0.005, 0.005],
    ]
    result = suggest(frames, 0, 4, "ab", 1)
    assert result["status"] == "ok"
    best = result["best"]
    assert best["cut_frame"] == 2
    assert best["probability"] == max(b["probability"] for b in result["boundaries"])
    assert best["probability"] >= CUT_THRESHOLD
    assert result["suitable"] is True
    assert "第 2 帧前" in result["message"]


def test_existing_repeat_char_requires_blank_and_is_exact():
    """``aa`` over frames a,a,a is unreachable; a,blank,a gives a real cut.

    This is the documented CTC rule: the repeated character only exists when a
    blank separates the two emissions.  The scorer must not treat the middle
    frames as "close enough".
    """
    no_blank = [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    unreachable = suggest(no_blank, 0, 2, "aa", 1)
    assert unreachable["status"] == "unreachable"
    assert unreachable["best"] is None
    assert unreachable["boundaries"] == []
    assert unreachable["full_text_logp"] is None

    with_blank = [
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
    ]
    z, numerators = brute_force_boundaries(with_blank, "aa", 1)
    reachable = suggest(with_blank, 0, 2, "aa", 1)
    assert reachable["status"] == "ok"
    for entry in reachable["boundaries"]:
        assert entry["probability"] == pytest.approx(
            numerators[entry["cut_frame"]] / z, abs=1e-12
        )
    # Only the middle boundary can be the cut: it is the only blank frame.
    assert reachable["best"]["cut_frame"] == 2


def test_zero_probability_paths_are_not_reachable():
    frames = [
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
    ]
    result = suggest(frames, 0, 1, "ba", 1)
    assert result["status"] == "unreachable"
    assert "没有任何 CTC 路径" in result["message"]


def test_unreachable_reports_reason_without_scores():
    frames = [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]
    result = suggest(frames, 0, 1, "ba", 1)
    assert result["status"] == "unreachable"
    assert result["full_text_logp"] is None
    assert result["suitable"] is False
    assert result["message"]


def test_all_zero_frame_is_reported_by_absolute_index():
    frames = [
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
    ]
    result = suggest(frames, 10, 12, "ab", 1)
    assert result["status"] == "all_zero"
    assert result["normalised"] is False
    assert result["zero_frame"] == 11
    assert "第 11 帧" in result["message"]
    assert result["boundaries"] == []
    assert result["best"] is None


def test_exact_tie_prefers_earliest_boundary():
    # A palindromic frame pattern makes cuts before frames 2 and 3 equally
    # probable.  The two values differ only in the last floating-point bits,
    # and the earlier boundary must still be chosen.
    frames = [
        [0.2, 0.4, 0.4],
        [0.4, 0.4, 0.2],
        [0.4, 0.2, 0.4],
        [0.2, 0.4, 0.4],
    ]
    result = suggest(frames, 0, 3, "ab", 1)
    assert result["status"] == "ok"
    probs = {entry["cut_frame"]: entry["probability"] for entry in result["boundaries"]}
    assert probs[2] == pytest.approx(probs[3], abs=1e-12)
    assert result["best"]["cut_frame"] == 2


def test_below_threshold_is_not_suitable_but_still_recommended():
    frames = [
        [0.34, 0.33, 0.33],
        [0.33, 0.34, 0.33],
        [0.33, 0.33, 0.34],
        [0.33, 0.34, 0.33],
    ]
    result = suggest(frames, 0, 3, "ab", 1)
    assert result["status"] == "ok"
    assert result["best"]["probability"] < CUT_THRESHOLD
    assert result["suitable"] is False
    assert "不宜" in result["message"]
    assert result["best"]["probability"] > 0.0


def test_all_boundaries_zero_for_reachable_text():
    """The text is reachable over the whole segment, but no cut works.

    "a" appears only in the first frame and "b" only in the last, so the
    required blank before a cut can never coincide with a boundary that keeps
    both halves decodable.
    """
    frames = [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = suggest(frames, 0, 2, "ab", 1)
    assert result["status"] == "ok"
    assert result["best"] is not None
    assert result["best"]["probability"] == 0.0
    assert result["suitable"] is False
    assert "均为 0" in result["message"]


# ------------------------------------------------------------------- validation


def test_segment_length_bounds():
    uniform = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    for length, ok in ((2, True), (64, True), (1, False), (65, False)):
        frames = np.asarray([uniform] * length)
        if ok:
            result = compute_cut_suggestion(frames, VOCAB, 0, length - 1, "ab", 1)
            assert result["status"] == "ok"
            assert len(result["boundaries"]) == length - 1
        else:
            with pytest.raises(DecodeError):
                compute_cut_suggestion(frames, VOCAB, 0, length - 1, "ab", 1)


def test_text_length_and_vocabulary_bounds():
    frames = np.asarray([[0.5, 0.5, 0.0]] * 4)
    with pytest.raises(DecodeError):
        compute_cut_suggestion(frames, VOCAB, 0, 3, "a", 1)
    with pytest.raises(DecodeError):
        compute_cut_suggestion(frames, VOCAB, 0, 3, "a" * 17, 1)
    with pytest.raises(DecodeError):
        compute_cut_suggestion(frames, VOCAB, 0, 3, "az", 1)
    with pytest.raises(DecodeError):
        compute_cut_suggestion(frames, VOCAB, 0, 3, "ab", 0)
    with pytest.raises(DecodeError):
        compute_cut_suggestion(frames, VOCAB, 0, 3, "ab", 2)


def test_normalise_frame_probs_reports_first_zero_row():
    probs = np.asarray([[0.5, 0.5], [0.0, 0.0], [0.2, 0.8]])
    normalised, zero_index = normalise_frame_probs(probs)
    assert normalised is None
    assert zero_index == 1
    ok, none = normalise_frame_probs(np.asarray([[2.0, 2.0]]))
    assert none is None
    assert ok is not None and ok.tolist() == [[0.5, 0.5]]


def test_validate_cut_inputs_returns_label_ids():
    ids = validate_cut_inputs(VOCAB, 0, 5, "abba", 2)
    assert ids == [1, 2, 2, 1]
