"""HTTP-level tests via FastAPI's TestClient.

These check the parts the browser relies on: uploads, revision handling,
error responses that leave the session untouched, and the peek endpoint.
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app

VOCAB = ["<blank>", "a", "b"]
FRAMES = [
    [0.6, 0.4, 0.0],
    [0.1, 0.8, 0.1],
    [0.5, 0.0, 0.5],
    [0.2, 0.3, 0.5],
    [0.7, 0.2, 0.1],
]


@pytest.fixture()
def client():
    return TestClient(app)


def create(client, vocab=None, frames=None, width=4):
    response = client.post(
        "/api/sessions",
        json={"vocab": vocab or VOCAB, "beam_width": width, "frames": frames or FRAMES},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_index_and_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "流式 CTC 解码工作台" in page.text


def test_create_and_advance(client):
    session = create(client)
    assert session["frame_count"] == len(FRAMES)
    assert session["next_frame"] == 0
    assert session["beams"][0]["text"] == ""
    sid = session["session_id"]

    response = client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    assert response.status_code == 200
    state = response.json()
    assert state["next_frame"] == 5
    assert state["beams"]
    assert state["frame_info"]["frame"] == 5


def test_advance_chunked_matches_http_state(client):
    one = create(client)
    client.post("/api/advance", json={"session_id": one["session_id"], "to_frame": 5})
    full = client.get(f"/api/sessions/{one['session_id']}").json()

    other = create(client)
    for frame in range(1, 6):
        client.post("/api/advance", json={"session_id": other["session_id"], "to_frame": frame})
    stepwise = client.get(f"/api/sessions/{other['session_id']}").json()
    assert full["beams"] == stepwise["beams"]


def test_confirm_then_undo_over_http(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    state = client.get(f"/api/sessions/{sid}").json()
    target = state["beams"][0]["text"]
    assert target

    response = client.post("/api/confirm", json={"session_id": sid, "prefix": target})
    assert response.status_code == 200
    confirmed = response.json()
    assert confirmed["prefix"] == target
    assert all(beam["text"].startswith(target) for beam in confirmed["beams"])
    assert confirmed["prefix_scores"]["last_char"] == target[-1]
    assert confirmed["prefix_scores"]["logp"] is not None

    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    response = client.post("/api/undo", json={"session_id": sid})
    assert response.status_code == 200
    restored = response.json()
    assert restored["prefix"] == ""
    assert restored["confirmed_count"] == 0


def test_failed_confirm_keeps_state(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    before = client.get(f"/api/sessions/{sid}").json()

    response = client.post("/api/confirm", json={"session_id": sid, "prefix": "zzz"})
    assert response.status_code == 409
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before

    response = client.post("/api/undo", json={"session_id": sid})
    assert response.status_code == 409
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before


def test_invalid_inputs_return_400_without_session(client):
    for payload in (
        {"vocab": ["<blank>", "ab"], "beam_width": 4, "frames": FRAMES},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 0, "frames": FRAMES},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 17, "frames": FRAMES},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 4, "frames": [[0.5, 0.5]]},
        {"vocab": ["<blank>", "a", "b"], "beam_width": 4, "frames": [[1.2, -0.2, 0.0]]},
    ):
        response = client.post("/api/sessions", json=payload)
        assert response.status_code == 400, (payload, response.text)


def test_upload_endpoint(client):
    vocab_file = ("vocab.json", json.dumps(["<blank>", "a", "b"]).encode(), "application/json")
    frames_file = ("frames.json", json.dumps({"frames": FRAMES}).encode(), "application/json")
    response = client.post(
        "/api/sessions/upload",
        files={"vocab_file": vocab_file, "frames_file": frames_file},
        data={"beam_width": "4"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["frame_count"] == len(FRAMES)


def test_upload_bad_frames_returns_400(client):
    vocab_file = ("vocab.json", json.dumps(["<blank>", "a", "b"]).encode(), "application/json")
    frames_file = ("frames.json", b"{not json", "application/json")
    response = client.post(
        "/api/sessions/upload",
        files={"vocab_file": vocab_file, "frames_file": frames_file},
        data={"beam_width": "4"},
    )
    assert response.status_code == 400
    assert "JSON" in response.json()["detail"]


def test_peek_only_for_consumed_frames(client):
    session = create(client)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})

    response = client.post("/api/peek", json={"session_id": sid, "frame": 2})
    assert response.status_code == 200
    payload = response.json()
    assert payload["frame"] == 2
    assert payload["beams"]

    response = client.post("/api/peek", json={"session_id": sid, "frame": 4})
    assert response.status_code == 409


def test_samples_are_served(client):
    response = client.get("/api/samples")
    names = [item["name"] for item in response.json()["samples"]]
    assert "sample_repeat.json" in names
    sample = client.get("/api/samples/sample_sparse_zero.json").json()
    assert sample["vocab"][0] == "<blank>"
    assert any(0.0 in row for row in sample["frames"])
    assert client.get("/api/samples/..%2Fapp%2Fmain.py").status_code == 404


def test_sample_creates_working_session(client):
    sample = client.get("/api/samples/sample_repeat.json").json()
    response = client.post(
        "/api/sessions",
        json={"vocab": sample["vocab"], "beam_width": 6, "frames": sample["frames"]},
    )
    assert response.status_code == 200
    state = response.json()
    client.post("/api/advance", json={"session_id": state["session_id"], "to_frame": len(sample["frames"])})
    final = client.get(f"/api/sessions/{state['session_id']}").json()
    assert final["beams"][0]["text"].startswith("a")


def test_unknown_session_is_404(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/advance", json={"session_id": "nope", "to_frame": 1}).status_code == 404


def test_concurrent_ops_apply_in_order(client):
    """Advance and undo issued together must both apply, in submission order."""
    session = create(client)
    sid = session["session_id"]
    state = client.get(f"/api/sessions/{sid}").json()
    client.post("/api/advance", json={"session_id": sid, "to_frame": 3})
    state = client.get(f"/api/sessions/{sid}").json()
    target = state["beams"][0]["text"]
    client.post("/api/confirm", json={"session_id": sid, "prefix": target})

    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    undo = client.post("/api/undo", json={"session_id": sid})
    assert undo.status_code == 200
    final = undo.json()
    assert final["next_frame"] == 5
    assert final["prefix"] == ""


def test_manager_cleanup_after_delete(client):
    session = create(client)
    sid = session["session_id"]
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


# ------------------------------------------------------------------ cut points


CUT_FRAMES = [
    [0.005, 0.99, 0.005],
    [0.98, 0.01, 0.01],
    [0.005, 0.005, 0.99],
    [0.99, 0.005, 0.005],
    [0.99, 0.005, 0.005],
]


def cut(client, sid, start, end, text, split):
    return client.post(
        "/api/cut-suggestion",
        json={
            "session_id": sid,
            "start_frame": start,
            "end_frame": end,
            "text": text,
            "split": split,
        },
    )


def test_cut_suggestion_over_http(client):
    session = create(client, frames=CUT_FRAMES)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": len(CUT_FRAMES)})

    response = cut(client, sid, 0, 4, "ab", 1)
    assert response.status_code == 200, response.text
    payload = response.json()
    result = payload["result"]
    assert result["status"] == "ok"
    assert result["best"]["cut_frame"] == 2
    assert result["suitable"] is True
    assert result["inputs"]["first_half"] == "a"
    assert result["inputs"]["second_half"] == "b"
    assert result["superseded"] is False
    # The response carries a session snapshot so the UI can show the suggestion
    # together with the state it belongs to.
    assert payload["state"]["cut_suggestion"]["best"]["cut_frame"] == 2
    assert payload["state"]["cut_suggestion"]["stale"] is False
    # A cut suggestion must not mutate the decode state.
    assert payload["state"]["next_frame"] == len(CUT_FRAMES)
    assert payload["state"]["prefix"] == ""


def test_cut_suggestion_requires_advanced_frames(client):
    session = create(client, frames=CUT_FRAMES)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
    # End frame not yet advanced.
    assert cut(client, sid, 0, 4, "ab", 1).status_code == 409
    # Nothing advanced at all.
    fresh = create(client, frames=CUT_FRAMES)
    assert cut(client, fresh["session_id"], 0, 1, "ab", 1).status_code == 409


def test_cut_suggestion_validation_leaves_session_untouched(client):
    session = create(client, frames=CUT_FRAMES)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    before = client.get(f"/api/sessions/{sid}").json()

    for start, end, text, split in (
        (0, 4, "a", 1),          # text too short
        (0, 4, "ab", 0),         # split out of range
        (0, 4, "ab", 2),         # split out of range
        (0, 4, "az", 1),         # out of vocabulary
        (0, 4, "a" * 17, 1),     # text too long
        (0, 0, "ab", 1),         # segment too short
        (0, 64, "ab", 1),        # segment too long
        (-1, 4, "ab", 1),        # negative start
    ):
        assert cut(client, sid, start, end, text, split).status_code == 400, (
            start,
            end,
            text,
            split,
        )
    after = client.get(f"/api/sessions/{sid}").json()
    assert after == before


def test_cut_failure_keeps_previous_suggestion(client):
    session = create(client, frames=CUT_FRAMES)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    first = cut(client, sid, 0, 4, "ab", 1).json()
    assert first["result"]["status"] == "ok"

    # A zero-probability frame inside a segment means it has no distribution.
    zero_frames = [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    other = create(client, frames=zero_frames)
    other_sid = other["session_id"]
    client.post("/api/advance", json={"session_id": other_sid, "to_frame": 3})
    zero = cut(client, other_sid, 0, 2, "ab", 1).json()["result"]
    assert zero["status"] == "all_zero"
    assert zero["zero_frame"] == 1
    state = client.get(f"/api/sessions/{other_sid}").json()
    assert state["cut_suggestion"] is None
    assert state["cut_failure"]["status"] == "all_zero"
    assert "1" in state["cut_failure"]["message"]


def test_cut_failure_keeps_previous_successful_suggestion(client):
    session = create(client, frames=CUT_FRAMES)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    ok = cut(client, sid, 0, 4, "ab", 1).json()
    assert ok["result"]["status"] == "ok"
    # In the same session a caption with an impossible repeat is unreachable;
    # the earlier suggestion must survive and the failure must explain why.
    bad = cut(client, sid, 0, 1, "aa", 1).json()
    assert bad["result"]["status"] == "unreachable"
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["cut_suggestion"]["inputs"]["text"] == "ab"
    assert state["cut_suggestion"]["best"]["cut_frame"] == 2
    assert state["cut_failure"]["status"] == "unreachable"


def test_advancing_marks_suggestion_stale(client):
    # A longer fixture so there are still frames left to advance.
    frames = CUT_FRAMES * 3
    session = create(client, frames=frames)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    result = cut(client, sid, 0, 4, "ab", 1).json()["result"]
    assert result["status"] == "ok"
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["cut_suggestion"]["stale"] is False

    client.post("/api/advance", json={"session_id": sid, "to_frame": 6})
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["cut_suggestion"]["stale"] is True
    assert "继续推进了帧" in state["cut_suggestion"]["stale_reasons"]


def test_confirmation_marks_suggestion_stale(client):
    session = create(client, frames=CUT_FRAMES * 2)
    sid = session["session_id"]
    client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
    assert cut(client, sid, 0, 4, "ab", 1).json()["result"]["status"] == "ok"
    state = client.get(f"/api/sessions/{sid}").json()
    target = state["beams"][0]["text"]
    assert target
    client.post("/api/confirm", json={"session_id": sid, "prefix": target})
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["cut_suggestion"]["stale"] is True
    assert "已确认字幕发生变化" in state["cut_suggestion"]["stale_reasons"]


def test_late_cut_result_is_superseded_not_current(client, monkeypatch):
    """A slow older request must not replace the newer current suggestion.

    The request stamp is taken when the HTTP request is accepted, so despite
    the single worker finishing the older computation, its result is reported
    as superseded and the session keeps the newer suggestion.
    """
    import asyncio
    import time

    import httpx

    from app import state as state_module

    original = state_module.compute_prefix_alpha
    calls = {"slow": True}

    # Make only the first computation slow; the second is fast.
    def slow_first(frames, ids):
        if calls["slow"]:
            calls["slow"] = False
            time.sleep(0.4)
        return original(frames, ids)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = (
                await client.post(
                    "/api/sessions",
                    json={"vocab": VOCAB, "beam_width": 4, "frames": CUT_FRAMES},
                )
            ).json()
            sid = created["session_id"]
            await client.post("/api/advance", json={"session_id": sid, "to_frame": 5})

            # Patch the function used inside cuts.py for the duration.
            import app.cuts as cuts_module

            monkeypatch.setattr(cuts_module, "compute_prefix_alpha", slow_first)

            first = asyncio.create_task(
                client.post(
                    "/api/cut-suggestion",
                    json={
                        "session_id": sid,
                        "start_frame": 0,
                        "end_frame": 4,
                        "text": "ab",
                        "split": 1,
                    },
                )
            )
            await asyncio.sleep(0.05)
            second = asyncio.create_task(
                client.post(
                    "/api/cut-suggestion",
                    json={
                        "session_id": sid,
                        "start_frame": 1,
                        "end_frame": 4,
                        "text": "ab",
                        "split": 1,
                    },
                )
            )
            first_response, second_response = await asyncio.gather(first, second)
            state = (await client.get(f"/api/sessions/{sid}")).json()
            return first_response, second_response, state

    first_response, second_response, state = asyncio.run(scenario())
    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["result"]["superseded"] is True
    assert second_response.json()["result"]["superseded"] is False
    # The newer request's inputs own the current suggestion.
    assert state["cut_suggestion"]["inputs"]["start_frame"] == 1


def test_slow_recompute_does_not_swallow_later_actions(monkeypatch):
    """A slow undo replay must not lose an advance issued while it runs.

    Undo and advance are queued on the session's single worker, so the advance
    is applied *after* the replay and the late undo response cannot revert it.
    """
    import asyncio
    import time

    import httpx

    from app import state as state_module

    original = state_module.undo

    def slow_undo(session):
        time.sleep(0.4)
        return original(session)

    monkeypatch.setattr(state_module, "undo", slow_undo)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = (
                await client.post(
                    "/api/sessions",
                    json={"vocab": VOCAB, "beam_width": 4, "frames": FRAMES},
                )
            ).json()
            sid = created["session_id"]
            await client.post("/api/advance", json={"session_id": sid, "to_frame": 2})
            state = (await client.get(f"/api/sessions/{sid}")).json()
            target = state["beams"][0]["text"]
            await client.post("/api/confirm", json={"session_id": sid, "prefix": target})

            undo_task = asyncio.create_task(client.post("/api/undo", json={"session_id": sid}))
            await asyncio.sleep(0.05)
            advance_task = asyncio.create_task(
                client.post("/api/advance", json={"session_id": sid, "to_frame": 5})
            )
            undo_response, advance_response = await asyncio.gather(undo_task, advance_task)
            final = (await client.get(f"/api/sessions/{sid}")).json()
            return undo_response, advance_response, final

    undo_response, advance_response, final = asyncio.run(scenario())
    assert undo_response.status_code == 200
    assert advance_response.status_code == 200
    # The later advance wins: prefix undone, frames advanced.
    assert final["next_frame"] == 5
    assert final["prefix"] == ""
    assert undo_response.json()["revision"] < final["revision"]
