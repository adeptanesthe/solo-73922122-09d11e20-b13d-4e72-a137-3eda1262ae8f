/* Streaming CTC workbench UI.
 *
 * Every server response carries a monotonically increasing revision.  The UI
 * remembers the revision it currently shows and ignores any response that is
 * older, so a slow advance or undo replay can never overwrite a newer
 * confirm / undo.  Mutating requests are additionally serialised in a small
 * client-side queue, and while a request is in flight the saved subtitle and
 * progress stay on screen until the answer arrives (or an error is shown).
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    connStatus: $("conn-status"),
    setupPanel: $("setup-panel"),
    vocabFile: $("vocab-file"),
    framesFile: $("frames-file"),
    beamWidth: $("beam-width"),
    beamOut: $("beam-out"),
    chunkSize: $("chunk-size"),
    chunkOut: $("chunk-out"),
    loadBtn: $("load-btn"),
    resetBtn: $("reset-btn"),
    setupHint: $("setup-hint"),
    setupError: $("setup-error"),
    sampleButtons: $("sample-buttons"),
    workspace: $("workspace"),
    progressBar: $("progress-bar"),
    progressText: $("progress-text"),
    confirmedText: $("confirmed-text"),
    pendingText: $("pending-text"),
    confirmBest: $("confirm-best"),
    undoBtn: $("undo-btn"),
    stepBtn: $("step-btn"),
    runBtn: $("run-btn"),
    restartBtn: $("restart-btn"),
    stopBtn: $("stop-btn"),
    prefixInput: $("prefix-input"),
    applyPrefix: $("apply-prefix"),
    psText: $("ps-text"),
    psBlank: $("ps-blank"),
    psNonblank: $("ps-nonblank"),
    psTotal: $("ps-total"),
    opError: $("op-error"),
    opNotice: $("op-notice"),
    beamBody: $("beam-body"),
    beamEmpty: $("beam-empty"),
    peekFrame: $("peek-frame"),
    peekFrameOut: $("peek-frame-out"),
    peekNote: $("peek-note"),
    frameProbs: $("frame-probs"),
    historyList: $("history-list"),
    historyEmpty: $("history-empty"),
    cutStart: $("cut-start"),
    cutEnd: $("cut-end"),
    cutText: $("cut-text"),
    cutSplit: $("cut-split"),
    cutBtn: $("cut-btn"),
    cutRangeHint: $("cut-range-hint"),
    cutSplitPreview: $("cut-split-preview"),
    cutResult: $("cut-result"),
    cutBest: $("cut-best"),
    cutBestFrame: $("cut-best-frame"),
    cutBestProb: $("cut-best-prob"),
    cutBestVerdict: $("cut-best-verdict"),
    cutMessage: $("cut-message"),
    cutStale: $("cut-stale"),
    cutBody: $("cut-body"),
    cutFailure: $("cut-failure"),
    cutEmpty: $("cut-empty"),
    busy: $("busy"),
    busyText: $("busy-text"),
  };

  const state = {
    sessionId: null,
    generation: 0,      // bumped whenever a new session replaces the old one
    revision: -1,       // revision of the state currently displayed
    snapshot: null,     // last state object accepted from the server
    busy: false,
    running: false,
    runAbort: false,
    pending: 0,
    samples: [],
    // Cut suggestions: the shown result is only accepted while it belongs to
    // the session and to the inputs currently typed in the form.  Each
    // keystroke or frame edit bumps ``cutSerial`` so a late reply cannot pose
    // as an answer to the current inputs.
    cutSerial: 0,
    cutInFlight: 0,
    cutExternal: null,   // suggestion kept by the server (last successful one)
    cutEndAuto: true,    // segment end follows the advanced range until edited
    // Behind the scenes: the last shown state is kept while a request is in
    // flight, so a failed operation leaves the subtitles and progress intact.
    peekLive: false,
  };

  // ---------------------------------------------------------------- helpers

  const fmt = (value) => {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    if (!Number.isFinite(value)) return "-∞";
    return value.toFixed(4);
  };

  const pct = (value) => {
    if (!Number.isFinite(value)) return "0";
    if (value >= 0.01) return (value * 100).toFixed(1) + "%";
    return value.toExponential(2);
  };

  function setConnection(kind, label) {
    els.connStatus.dataset.state = kind;
    els.connStatus.textContent = label;
  }

  function setBusy(on, text = "计算中…") {
    state.busy = on;
    els.busy.hidden = !on;
    els.busyText.textContent = text;
    if (on) setConnection("busy", "计算中");
    else updateStatusFromSnapshot();
    updateButtons();
  }

  // Mark an operation as in flight.  Operations are *not* blocked while
  // another one runs: the server applies them in submission order and each
  // response is revision-guarded, so advancing / undoing during a replay is
  // safe and the late replay result cannot overwrite the newer action.
  function beginOp(label) {
    state.pending += 1;
    setBusy(true, label);
  }

  function endOp() {
    state.pending = Math.max(0, state.pending - 1);
    if (state.pending === 0) setBusy(false);
  }

  function updateStatusFromSnapshot() {
    if (!state.snapshot) { setConnection("idle", "未加载"); return; }
    if (state.running) setConnection("busy", "连续推进中");
    else setConnection("ready", "就绪");
  }

  function showError(message) {
    hideNotice();
    if (!message) { els.opError.hidden = true; els.opError.textContent = ""; return; }
    els.opError.hidden = false;
    els.opError.textContent = message;
  }

  function showNotice(message) {
    if (!message) { els.opNotice.hidden = true; els.opNotice.textContent = ""; return; }
    els.opNotice.hidden = false;
    els.opNotice.textContent = message;
  }

  function hideNotice() { els.opNotice.hidden = true; els.opNotice.textContent = ""; }

  function updateButtons() {
    const ready = !!state.sessionId && !!state.snapshot;
    const atEnd = ready && state.snapshot.next_frame >= state.snapshot.frame_count;
    // Recompute (undo replay / long advance) deliberately does NOT disable
    // advance or undo: those actions are queued by the server behind the
    // running operation and applied to its result, in order.
    els.confirmBest.disabled = !ready;
    els.undoBtn.disabled = !ready || (state.snapshot.confirmed_count || 0) === 0;
    els.stepBtn.disabled = !ready || atEnd || state.running;
    els.runBtn.disabled = !ready || atEnd || state.running;
    els.restartBtn.disabled = !ready || state.running;
    els.applyPrefix.disabled = !ready;
    els.stopBtn.hidden = !state.running;
    els.resetBtn.disabled = !ready;
    updateCutButtons();
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let payload = null;
    try { payload = await response.json(); } catch { payload = null; }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = response.status;
      throw err;
    }
    return payload;
  }

  // Apply a server state snapshot, unless it belongs to a replaced session or
  // is older than the state already displayed.  Together with the server's
  // in-order worker this is what makes late results harmless.
  function applyState(payload, { force = false } = {}) {
    if (!payload || typeof payload.revision !== "number") return false;
    if (payload.session_id !== state.sessionId) return false;
    if (!force && payload.revision < state.revision) return false;
    state.revision = payload.revision;
    state.snapshot = payload;
    render();
    return true;
  }

  function startSession(payload, sampleName = null) {
    // Invalidate anything still in flight for the previous session.
    state.generation += 1;
    state.runAbort = true;
    state.running = false;
    state.pending = 0;
    els.stopBtn.hidden = true;
    state.sessionId = payload.session_id;
    state.revision = -1;
    state.snapshot = null;
    state.loadedSample = sampleName;
    state.peekLive = true;
    // A new session invalidates every cut suggestion and any reply in flight.
    state.cutSerial += 1;
    state.cutInFlight = 0;
    state.cutExternal = null;
    els.cutResult.hidden = true;
    els.cutEmpty.hidden = false;
    els.cutEmpty.textContent = "尚未计算切点建议。";
    els.cutFailure.hidden = true;
    els.cutText.value = "";
    els.cutSplit.value = "1";
    els.cutStart.value = "0";
    state.cutEndAuto = true;
    els.cutEnd.value = "1";
    applyState(payload, { force: true });
    if (state.snapshot) {
      els.peekFrame.value = "0";
      els.peekFrame.max = String(state.snapshot.frame_count);
      els.peekFrameOut.textContent = `0 / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
    }
    updateButtons();
    updateStatusFromSnapshot();
  }

  // ---------------------------------------------------------------- rendering

  function render() {
    const snap = state.snapshot;
    if (!snap) { els.workspace.hidden = true; return; }
    els.workspace.hidden = false;

    const total = snap.frame_count || 0;
    const done = snap.next_frame || 0;
    els.progressBar.style.width = total ? `${(100 * done) / total}%` : "0%";
    els.progressText.textContent = `${done} / ${total} 帧`;

    const best = snap.best_text || "";
    const prefix = snap.prefix || "";
    const pending = best.startsWith(prefix) ? best.slice(prefix.length) : best;
    els.confirmedText.textContent = prefix;
    els.pendingText.textContent = pending;

    const scores = snap.prefix_scores || {};
    els.psText.textContent = scores.text ? `“${scores.text}” → ${scores.last_char}` : "—";
    els.psBlank.textContent = fmt(scores.logp_blank);
    els.psNonblank.textContent = fmt(scores.logp_nonblank);
    els.psBlank.title = "确认时的末字 blank 路径概率（保留值，不随新帧归一化）";
    els.psNonblank.title = "确认时的末字非 blank 路径概率（保留值，不随新帧归一化）";
    els.psTotal.textContent = fmt(scores.logp);
    if (scores.text) {
      els.psTotal.title =
        `确认于第 ${scores.confirmed_at} 帧；此前缀“${scores.text}”的完整前向分数 ` +
        `为 ${fmt(scores.logp)}，其中末字前的前缀分数为 ${fmt(scores.previous_logp)}。` +
        "值保留自确认时刻，不随后续帧或再次确认归一化。";
    } else {
      els.psTotal.title = "尚未确认前缀";
    }

    renderBeams(snap, prefix);

    els.peekFrame.max = String(total);
    if (Number(els.peekFrame.value) > total) els.peekFrame.value = String(total);
    els.peekFrameOut.textContent = `${els.peekFrame.value} / ${total}`;
    els.peekNote.textContent =
      Number(els.peekFrame.value) === done ? "显示当前帧" : "历史回放（只读）";

    renderHistory(snap);
    renderFrameProbs(snap.frame_info);
    els.prefixInput.placeholder = prefix ? `当前前缀：${prefix}` : "例如：啊";
    renderCut(snap);
    updateButtons();
    updateStatusFromSnapshot();
  }

  function renderCut(snap) {
    updateCutRangeHint();
    // While the end field has not been edited by hand, keep it on the last
    // advanced frame so "current segment" stays meaningful as frames arrive.
    if (state.cutEndAuto) {
      const liveEnd = Math.max(1, snap.next_frame - 1);
      els.cutEnd.value = String(liveEnd);
    }
    updateCutSplitPreview();
    // The server keeps the last successful suggestion (and the reason a later
    // attempt failed).  Adopt it unless this tab already holds a newer one.
    adoptServerCutSuggestion(snap);
    // Always re-render: advancing, confirming or editing an input must mark
    // the displayed suggestion as stale even without a new computation.
    if (state.cutExternal) renderCutSuggestion(state.cutExternal);
    else renderCutSuggestion(null);
    renderCutFailure(snap.cut_failure);
  }

  function renderBeams(snap, prefix) {
    const beams = snap.beams || [];
    if (typeof snap.revision === "number") state.peekLive = true;
    els.beamBody.innerHTML = "";
    els.beamEmpty.hidden = beams.length > 0;
    beams.forEach((beam, index) => {
      const tr = document.createElement("tr");
      if (index === 0) tr.className = "selected";

      const idx = document.createElement("td");
      idx.textContent = String(index + 1);

      const textTd = document.createElement("td");
      const span = document.createElement("span");
      span.className = "beam-text";
      const rest = beam.text.startsWith(prefix) ? beam.text.slice(prefix.length) : beam.text;
      if (prefix) {
        const pfx = document.createElement("span");
        pfx.className = "pfx";
        pfx.textContent = prefix;
        span.appendChild(pfx);
      }
      const tail = document.createElement("span");
      if (rest) tail.textContent = rest;
      else { tail.className = "empty-tag"; tail.textContent = "∅"; }
      span.appendChild(tail);
      textTd.appendChild(span);
      if (beam.text.startsWith(prefix)) textTd.title = beam.text;

      const logpTd = document.createElement("td");
      logpTd.className = "num";
      logpTd.textContent = fmt(beam.logp);

      const probTd = document.createElement("td");
      probTd.className = "num";
      probTd.textContent = pct(beam.prob);

      const blankTd = document.createElement("td");
      blankTd.className = "num";
      blankTd.textContent = fmt(beam.logp_blank);

      const nbTd = document.createElement("td");
      nbTd.className = "num";
      nbTd.textContent = fmt(beam.logp_nonblank);

      const actTd = document.createElement("td");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "确认此候选";
      btn.title = `确认前缀：${beam.text}`;
      btn.disabled = !state.peekLive;
      btn.addEventListener("click", () => confirmPrefix(beam.text));
      actTd.appendChild(btn);

      tr.append(idx, textTd, logpTd, probTd, blankTd, nbTd, actTd);
      els.beamBody.appendChild(tr);
    });
  }

  function renderHistory(snap) {
    const list = snap.confirmations || [];
    els.historyList.innerHTML = "";
    els.historyEmpty.hidden = list.length > 0;
    list.forEach((entry, index) => {
      const li = document.createElement("li");
      const left = document.createElement("span");
      const badge = document.createElement("span");
      badge.className = "h-meta";
      badge.textContent = `#${entry.seq} @${entry.frame} 帧`;
      const prefixSpan = document.createElement("span");
      prefixSpan.className = "h-prefix";
      prefixSpan.textContent = entry.prefix_after || "(空)";
      left.append(badge, document.createTextNode("  "), prefixSpan);

      const right = document.createElement("span");
      right.className = "h-meta";
      right.textContent =
        `logP ${fmt(entry.confirmed_logp)} · blank ${fmt(entry.confirmed_logp_blank)}` +
        ` · 非 blank ${fmt(entry.confirmed_logp_nonblank)}`;

      li.append(left, right);
      if (index === list.length - 1) li.style.borderColor = "rgba(79, 209, 197, .45)";
      els.historyList.appendChild(li);
    });
  }

  function renderFrameProbs(info) {
    els.frameProbs.innerHTML = "";
    if (!info || !info.probs) return;
    const vocab = info.vocab || [];
    info.probs.forEach((value, index) => {
      const cell = document.createElement("div");
      cell.className = "fp" + (value === 0 ? " zero" : "");
      const tok = document.createElement("span");
      tok.className = "tok";
      tok.textContent = index === 0 ? "<blank>" : vocab[index];
      const val = document.createElement("span");
      val.className = "val";
      val.textContent = "  " + value.toFixed(3);
      cell.append(tok, val);
      cell.title = `log P = ${info.log_probs && info.log_probs[index] !== null ? Number(info.log_probs[index]).toFixed(4) : "-∞"}`;
      els.frameProbs.appendChild(cell);
    });
  }

  // -------------------------------------------------------- cut suggestions

  // The inputs a suggestion belongs to, in the same normalised form the
  // server echoes back.  Comparing these makes local edits (segment, text or
  // split) invalidate the displayed suggestion immediately.
  function readCutInputs() {
    const start = els.cutStart.value.trim();
    const end = els.cutEnd.value.trim();
    const split = els.cutSplit.value.trim();
    return {
      start_frame: start === "" ? null : Number(start),
      end_frame: end === "" ? null : Number(end),
      text: els.cutText.value.trim(),
      split: split === "" ? null : Number(split),
    };
  }

  function sameCutInputs(a, b) {
    if (!a || !b) return false;
    return a.start_frame === b.start_frame
      && a.end_frame === b.end_frame
      && a.text === b.text
      && a.split === b.split;
  }

  // A shown result is "current" only if it was computed for the inputs that
  // are in the form right now.  Anything else is stale.
  function cutIsStale(result) {
    if (!result) return true;
    if (result.stale) return true;   // verdict recorded by the server
    const live = readCutInputs();
    if (!result.inputs) return true;
    if (!sameCutInputs(result.inputs, live)) return true;
    if (state.snapshot && result.computed_at
        && result.computed_at.next_frame !== state.snapshot.next_frame) return true;
    if (state.snapshot && result.computed_at
        && result.computed_at.prefix !== (state.snapshot.prefix || "")) return true;
    return false;
  }

  function staleReasonLabel(result) {
    const live = readCutInputs();
    const inputs = (result && result.inputs) || {};
    const reasons = [];
    if (inputs.start_frame !== live.start_frame || inputs.end_frame !== live.end_frame) {
      reasons.push("片段已更改");
    }
    if (inputs.text !== live.text) reasons.push("字幕已更改");
    if (inputs.split !== live.split) reasons.push("切分处已更改");
    if (state.snapshot && result && result.computed_at) {
      if (result.computed_at.next_frame !== state.snapshot.next_frame) reasons.push("已继续推进帧");
      if (result.computed_at.prefix !== (state.snapshot.prefix || "")) reasons.push("已确认字幕已变化");
    }
    (result && result.stale_reasons ? result.stale_reasons : []).forEach((reason) => {
      if (!reasons.includes(reason)) reasons.push(reason);
    });
    return reasons;
  }

  function updateCutButtons() {
    // Deliberately stays enabled while a computation runs: submitting newer
    // inputs must supersede the running request (the server stamps requests
    // on arrival and labels the older result ``superseded``).
    const ready = !!state.sessionId && !!state.snapshot;
    const enough = ready && state.snapshot.next_frame >= 2;
    els.cutBtn.disabled = !enough;
    els.cutBtn.title = enough ? "" : "至少推进 2 帧后才能计算切点";
    els.cutBtn.textContent = state.cutInFlight > 0 ? "重新计算切点（有计算进行中）" : "计算切点建议";
  }

  function updateCutRangeHint() {
    const snap = state.snapshot;
    if (!snap) {
      els.cutRangeHint.textContent = "片段须在已推进的帧内，2～64 帧";
      return;
    }
    const advanceable = snap.next_frame - 1;
    els.cutRangeHint.textContent = advanceable >= 1
      ? `可用帧 0～${advanceable}（已推进 ${snap.next_frame} 帧，片段 2～64 帧）`
      : "至少推进 2 帧后才能选择片段";
  }

  function updateCutSplitPreview() {
    const text = els.cutText.value.trim();
    const splitValue = Number(els.cutSplit.value);
    if (!text) { els.cutSplitPreview.textContent = "—"; return; }
    if (!Number.isInteger(splitValue) || splitValue < 1 || splitValue >= text.length) {
      els.cutSplitPreview.textContent = `切分处须在 1～${Math.max(0, text.length - 1)} 之间`;
      return;
    }
    els.cutSplitPreview.textContent = `左：“${text.slice(0, splitValue)}”　右：“${text.slice(splitValue)}”`;
  }

  function renderCutFailure(failure) {
    if (!failure || !failure.message) {
      els.cutFailure.hidden = true;
      els.cutFailure.textContent = "";
      return;
    }
    els.cutFailure.hidden = false;
    els.cutFailure.textContent = failure.message;
  }

  function renderCutSuggestion(result) {
    if (!result) {
      els.cutResult.hidden = true;
      els.cutEmpty.hidden = false;
      els.cutEmpty.textContent = "尚未计算切点建议。";
      return;
    }
    els.cutResult.hidden = false;
    els.cutEmpty.hidden = true;
    const stale = cutIsStale(result);
    const best = result.best;
    const inputs = result.inputs || {};
    // A suggestion whose status is not "ok" has no cut to recommend; it is
    // still displayed (with its explanation) but never as usable.
    const suitable = result.status === "ok" && !!best && !!result.suitable;
    els.cutBest.dataset.verdict = suitable ? "good" : "bad";
    if (best) {
      els.cutBestFrame.textContent = `第 ${best.cut_frame} 帧前`;
      els.cutBestFrame.title =
        `左侧 ${inputs.start_frame}～${best.prev_frame} 帧（“${inputs.first_half}”），` +
        `右侧 ${best.cut_frame}～${inputs.end_frame} 帧（“${inputs.second_half}”）`;
      els.cutBestProb.textContent = best.probability.toFixed(4);
      els.cutBestProb.title =
        `P(切于此 | 片段折叠为全文) = ${best.probability.toFixed(6)}；` +
        `log = ${fmt(best.log_probability)}。阈值 ${result.threshold}。`;
      els.cutBestVerdict.textContent = suitable ? "可以切开" : "不宜切开";
      els.cutBestVerdict.title = suitable
        ? `条件概率不低于阈值 ${result.threshold}`
        : `最高条件概率低于阈值 ${result.threshold}`;
    } else {
      els.cutBestFrame.textContent = "—";
      els.cutBestProb.textContent = "—";
      els.cutBestVerdict.textContent = "不宜切开";
    }
    els.cutMessage.textContent = result.message || "";

    const reasons = stale ? staleReasonLabel(result) : [];
    els.cutStale.hidden = !stale;
    els.cutStale.textContent = stale
      ? `此建议已过期：${reasons.length ? reasons.join("、") : "输入已变化"}。` +
        `它属于片段 ${inputs.start_frame}～${inputs.end_frame} 帧、字幕“${inputs.text}”、` +
        `切分处 ${inputs.split}${result.computed_at ? `、第 ${result.computed_at.next_frame} 帧时` : ""}。` +
        "请重新计算。"
      : "";

    els.cutBody.innerHTML = "";
    (result.boundaries || []).forEach((entry) => {
      const tr = document.createElement("tr");
      if (best && entry.cut_frame === best.cut_frame) tr.className = "recommended";
      if (!entry.probability) tr.classList.add("zero");
      const frameTd = document.createElement("td");
      frameTd.textContent = String(entry.cut_frame);
      const leftTd = document.createElement("td");
      leftTd.className = "num";
      leftTd.textContent = `${inputs.start_frame}～${entry.prev_frame}`;
      const probTd = document.createElement("td");
      probTd.className = "num";
      probTd.textContent = entry.probability.toFixed(6);
      const logTd = document.createElement("td");
      logTd.className = "num";
      logTd.textContent = fmt(entry.log_probability);
      tr.append(frameTd, leftTd, probTd, logTd);
      els.cutBody.appendChild(tr);
    });
  }

  // Apply a suggestion stored on a session snapshot.  It is only a fallback:
  // a live result received by this tab always wins while its inputs match.
  // Callers render the adopted suggestion; this function only updates state.
  function adoptServerCutSuggestion(snap) {
    const suggestion = snap.cut_suggestion;
    if (!suggestion) { state.cutExternal = null; return; }
    if (state.cutExternal && state.cutExternal.order >= suggestion.order) return;
    state.cutExternal = suggestion;
  }

  async function requestCutSuggestion() {
    if (!state.sessionId || !state.snapshot) return;
    const live = readCutInputs();
    if (live.start_frame === null || live.end_frame === null || live.split === null) {
      showError("请填写片段起始帧、结束帧与切分处");
      return;
    }
    if (!Number.isInteger(live.start_frame) || !Number.isInteger(live.end_frame)
        || !Number.isInteger(live.split)) {
      showError("帧号与切分处必须是整数");
      return;
    }
    const sessionId = state.sessionId;
    const generation = state.generation;
    state.cutSerial += 1;
    const serial = state.cutSerial;
    state.cutInFlight += 1;
    updateCutButtons();
    try {
      const payload = await api("/api/cut-suggestion", {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          start_frame: live.start_frame,
          end_frame: live.end_frame,
          text: live.text,
          split: live.split,
        }),
      });
      if (generation !== state.generation || sessionId !== state.sessionId) return;
      const result = payload.result;
      // A reply is only shown if it is the newest request *and* the form has
      // not been edited since it was sent.  A late reply for older inputs is
      // reported but never displayed as the current suggestion.
      if (result.superseded || serial !== state.cutSerial
          || !sameCutInputs(result.inputs, readCutInputs())) {
        showNotice("切点计算结果已过期（输入或会话已变化），未采用。");
        if (payload.state) {
          applyState(payload.state);
          if (payload.state.cut_suggestion) {
            state.cutExternal = payload.state.cut_suggestion;
            renderCutSuggestion(state.cutExternal);
          }
        }
        return;
      }
      if (payload.state) applyState(payload.state);
      if (result.status === "ok" && result.best) {
        state.cutExternal = { ...result };
        // ``result`` already recorded the exact inputs and session position it
        // was computed for, so the staleness check will agree with itself.
        renderCutSuggestion(result);
        renderCutFailure(null);
        showError("");
        hideNotice();
      } else {
        // No usable suggestion: the server kept the previous one (and echoed
        // it in ``state.cut_suggestion``).  Keep showing that, never the
        // failed attempt, and explain the failure in its own line.
        if (payload.state && payload.state.cut_suggestion) {
          state.cutExternal = payload.state.cut_suggestion;
          renderCutSuggestion(state.cutExternal);
        } else {
          renderCutSuggestion(null);
        }
        renderCutFailure(payload.state ? payload.state.cut_failure : {
          message: result.message,
          inputs: result.inputs,
        });
        showError("");
        hideNotice();
      }
    } catch (error) {
      if (generation !== state.generation || sessionId !== state.sessionId) return;
      showError(`切点建议失败：${error.message}（旧建议与输入保持不变）`);
      renderCutFailure({
        message: `本次尝试未产生建议：${error.message}`,
        inputs: live,
      });
      if (state.cutExternal) renderCutSuggestion(state.cutExternal);
    } finally {
      if (generation === state.generation) {
        state.cutInFlight = Math.max(0, state.cutInFlight - 1);
        updateCutButtons();
      }
    }
  }

  // ------------------------------------------------------------- operations

  async function withBusy(label, fn) {
    beginOp(label);
    try {
      return await fn();
    } finally {
      endOp();
    }
  }

  async function createSession(formData) {
    els.setupError.hidden = true;
    setBusy(true, "解析并创建会话…");
    try {
      const payload = await api("/api/sessions/upload", { method: "POST", body: formData, headers: {} });
      startSession(payload);
      showError("");
      hideNotice();
      els.loadBtn.disabled = false;
      setConnection("ready", "就绪");
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
      els.loadBtn.disabled = false;
    }
  }

  async function advanceTo(target, { silentBusy = false } = {}) {
    if (!state.sessionId) return false;
    const sessionId = state.sessionId;
    const generation = state.generation;
    const run = async () => {
      try {
        const payload = await api("/api/advance", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            to_frame: target,
          }),
        });
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        const applied = applyState(payload);
        if (applied) { showError(""); hideNotice(); }
        return applied;
      } catch (error) {
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        showError(`推进失败：${error.message}（字幕与进度保持不变）`);
        return false;
      }
    };
    return silentBusy ? run() : withBusy("推进帧…", run);
  }

  async function confirmPrefix(prefix) {
    if (!state.sessionId) return;
    if (!prefix || prefix === (state.snapshot && state.snapshot.prefix)) {
      showError("请输入比当前已确认前缀更长的文本");
      return;
    }
    if (!state.peekLive) {
      showError("正在查看历史帧；请先把查看帧拖回当前帧再确认");
      return;
    }
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("确认前缀…", async () => {
      try {
        const payload = await api("/api/confirm", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            prefix,
          }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice(`已确认前缀「${prefix}」。后续只保留兼容候选。`);
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`确认失败：${error.message}（前缀与候选保持不变）`);
      }
    });
  }

  async function undo() {
    if (!state.sessionId) return;
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("撤销确认并重算…", async () => {
      try {
        const payload = await api("/api/undo", {
          method: "POST",
          body: JSON.stringify({ session_id: sessionId }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice("已撤销上次确认，束与分数恢复到未作该次确认时应有的状态。");
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`撤销失败：${error.message}（字幕与进度保持不变）`);
      }
    });
  }

  async function peek(frame) {
    if (!state.sessionId) return;
    try {
      const payload = await api("/api/peek", {
        method: "POST",
        body: JSON.stringify({ session_id: state.sessionId, frame }),
      });
      // Peek responses are informational only; they must not clobber the live
      // beam table while newer state is displayed.
      if (payload.revision !== state.revision) return;
      if (!state.snapshot) return;
      if (frame === state.snapshot.next_frame) {
        // Back on the live boundary: show the live state, not a replay.
        state.peekLive = true;
        renderBeams(state.snapshot, state.snapshot.prefix || "");
        renderFrameProbs(state.snapshot.frame_info);
        els.peekNote.textContent = "显示当前帧";
        return;
      }
      const prefix = payload.prefix || "";
      renderBeams({ beams: payload.beams, prefix }, prefix);
      state.peekLive = false;
      renderFrameProbs(payload.frame_info);
      els.peekNote.textContent = `历史回放：第 ${frame} 帧（只读，确认按钮无效）`;
    } catch (error) {
      if (frame === (state.snapshot ? state.snapshot.next_frame : -1)) {
        showError(`查看帧失败：${error.message}`);
      }
    }
  }

  async function runContinuously() {
    if (!state.sessionId || !state.snapshot || state.running) return;
    state.running = true;
    state.runAbort = false;
    state.busy = false;
    updateButtons();
    setConnection("busy", "连续推进中");
    while (state.running && !state.runAbort) {
      const snap = state.snapshot;
      if (!snap || snap.next_frame >= snap.frame_count) break;
      const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
      const target = Math.min(snap.frame_count, snap.next_frame + chunk);
      // Advance without the modal busy overlay; the status pill shows progress.
      const ok = await advanceTo(target, { silentBusy: true });
      if (!ok) break;
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent =
        `${state.snapshot.next_frame} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  }

  // ------------------------------------------------------------------- wiring

  els.beamWidth.addEventListener("input", () => {
    els.beamOut.textContent = els.beamWidth.value;
  });
  els.chunkSize.addEventListener("input", () => {
    els.chunkOut.textContent = els.chunkSize.value;
  });

  els.loadBtn.addEventListener("click", () => {
    const vocabFile = els.vocabFile.files[0];
    const framesFile = els.framesFile.files[0];
    if (!vocabFile || !framesFile) {
      els.setupError.hidden = false;
      els.setupError.textContent = "请同时选择词表文件和逐帧概率文件";
      return;
    }
    const form = new FormData();
    form.append("vocab_file", vocabFile);
    form.append("frames_file", framesFile);
    form.append("beam_width", els.beamWidth.value);
    createSession(form);
  });

  els.resetBtn.addEventListener("click", async () => {
    if (state.sessionId) {
      try {
        await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
      } catch { /* ignore */ }
    }
    state.sessionId = null;
    state.revision = -1;
    state.snapshot = null;
    state.cutSerial += 1;
    state.cutInFlight = 0;
    state.cutExternal = null;
    els.workspace.hidden = true;
    state.loadedSample = null;
    setConnection("idle", "未加载");
    updateButtons();
  });

  els.confirmBest.addEventListener("click", () => {
    const snap = state.snapshot;
    if (!snap) return;
    const best = snap.best_text || "";
    let target = best;
    if (!best || best === snap.prefix) {
      showError("当前最佳候选与已确认前缀相同，无法继续确认");
      return;
    }
    // confirm the full best hypothesis; users can also type a shorter prefix
    confirmPrefix(target);
  });

  els.applyPrefix.addEventListener("click", () => {
    confirmPrefix(els.prefixInput.value.trim());
  });

  els.prefixInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      confirmPrefix(els.prefixInput.value.trim());
    }
  });

  els.undoBtn.addEventListener("click", undo);

  // Cut-point inputs.  Editing any of them bumps ``cutSerial`` so a reply that
  // is still in flight can no longer be displayed as the current suggestion;
  // the old result stays visible but renderCutSuggestion marks it stale.
  els.cutStart.addEventListener("input", () => {
    state.cutSerial += 1;
    updateCutSplitPreview();
    if (state.cutExternal) renderCutSuggestion(state.cutExternal);
  });
  els.cutEnd.addEventListener("input", () => {
    state.cutEndAuto = false;
    state.cutSerial += 1;
    updateCutSplitPreview();
    if (state.cutExternal) renderCutSuggestion(state.cutExternal);
  });
  els.cutSplit.addEventListener("input", () => {
    state.cutSerial += 1;
    updateCutSplitPreview();
    if (state.cutExternal) renderCutSuggestion(state.cutExternal);
  });
  els.cutText.addEventListener("input", () => {
    state.cutSerial += 1;
    updateCutSplitPreview();
    if (state.cutExternal) renderCutSuggestion(state.cutExternal);
  });
  els.cutBtn.addEventListener("click", requestCutSuggestion);

  els.stepBtn.addEventListener("click", async () => {
    const snap = state.snapshot;
    if (!snap) return;
    const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
    const target = Math.min(snap.frame_count, snap.next_frame + chunk);
    const ok = await advanceTo(target);
    if (ok) {
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent = `${els.peekFrame.value} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      renderFrameProbs(state.snapshot.frame_info);
    }
  });

  els.runBtn.addEventListener("click", runContinuously);
  els.stopBtn.addEventListener("click", () => {
    state.runAbort = true;
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  });

  els.restartBtn.addEventListener("click", async () => {
    // Reset by opening a fresh server session from the same sample or files.
    state.runAbort = true;
    if (state.loadedSample) {
      await loadSample(state.loadedSample);
      return;
    }
    if (!els.vocabFile.files[0] || !els.framesFile.files[0]) {
      showError("要回到起点请重新选择文件或重新载入样例");
      return;
    }
    const form = new FormData();
    form.append("vocab_file", els.vocabFile.files[0]);
    form.append("frames_file", els.framesFile.files[0]);
    form.append("beam_width", els.beamWidth.value);
    await createSession(form);
  });

  els.peekFrame.addEventListener("input", () => {
    const frame = Number(els.peekFrame.value);
    els.peekFrameOut.textContent = `${frame} / ${els.peekFrame.max}`;
    peek(frame);
  });

  async function loadSamples() {
    try {
      const payload = await api("/api/samples");
      state.samples = payload.samples || [];
      els.sampleButtons.innerHTML = "";
      if (!state.samples.length) {
        const span = document.createElement("span");
        span.className = "hint";
        span.textContent = "未找到样例文件";
        els.sampleButtons.appendChild(span);
        return;
      }
      state.samples.forEach((sample) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample ghost";
        btn.textContent = sample.name;
        btn.addEventListener("click", () => loadSample(sample.name));
        els.sampleButtons.appendChild(btn);
      });
    } catch { /* samples are optional */ }
  }

  async function loadSample(name) {
    els.setupError.hidden = true;
    setBusy(true, "载入样例…");
    try {
      const sample = await api(`/api/samples/${encodeURIComponent(name)}`);
      const payload = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          vocab: sample.vocab,
          beam_width: Number(els.beamWidth.value),
          frames: sample.frames,
        }),
      });
      startSession(payload, name);
      showError("");
      hideNotice();
      showNotice(`已载入样例「${name}」，共 ${payload.frame_count} 帧。`);
      // Prefill from the best hypothesis so the first attempt needs one click;
      // the user can still edit every field.
      const best = payload.best_text || "";
      if (best.length >= 2) {
        els.cutText.value = best;
        els.cutSplit.value = "1";
      }
      render();
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
    }
  }

  loadSamples();
  updateButtons();
  updateStatusFromSnapshot();
})();
