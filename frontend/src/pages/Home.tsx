import { useCallback, useEffect, useRef, useState } from "react";
import { startSimulation, fetchOutcome, fetchProspectTurn } from "../api/client";
import { OutcomeView } from "../components/OutcomeView";
import type { Outcome, TurnResponse } from "../types/outcome";

type CallStatus = "idle" | "starting" | "connecting" | "in-call" | "ended";
type ProspectMode = "human" | "ai";

interface TurnLog {
  role: "prospect" | "agent";
  text: string;
}

// ---------------------------------------------------------------------------
// Minimal protobuf encode/decode for pipecat Frame messages
// ---------------------------------------------------------------------------

function encodeVarint(value: number): Uint8Array {
  const bytes: number[] = [];
  do {
    let byte = value & 0x7f;
    value >>>= 7;
    if (value > 0) byte |= 0x80;
    bytes.push(byte);
  } while (value > 0);
  return new Uint8Array(bytes);
}

function decodeVarint(buf: Uint8Array, offset: number): [number, number] {
  let result = 0;
  let shift = 0;
  let pos = offset;
  while (pos < buf.length) {
    const byte = buf[pos];
    result |= (byte & 0x7f) << shift;
    pos++;
    if ((byte & 0x80) === 0) break;
    shift += 7;
  }
  return [result, pos];
}

/** Encode PCM16 audio into a pipecat Frame protobuf binary message. */
function encodeAudioFrame(
  pcm16: ArrayBuffer,
  sampleRate: number,
  numChannels: number,
): Uint8Array {
  const audioData = new Uint8Array(pcm16);
  // AudioRawFrame fields: audio(3,bytes), sample_rate(4,uint32), num_channels(5,uint32)
  const f3tag = new Uint8Array([0x1a]); // field 3, wire type 2
  const f3len = encodeVarint(audioData.length);
  const f4tag = new Uint8Array([0x20]); // field 4, wire type 0
  const f4val = encodeVarint(sampleRate);
  const f5tag = new Uint8Array([0x28]); // field 5, wire type 0
  const f5val = encodeVarint(numChannels);

  const innerLen =
    f3tag.length + f3len.length + audioData.length +
    f4tag.length + f4val.length +
    f5tag.length + f5val.length;

  // Frame.audio = field 2, wire type 2
  const outerTag = new Uint8Array([0x12]);
  const outerLen = encodeVarint(innerLen);

  const out = new Uint8Array(outerTag.length + outerLen.length + innerLen);
  let p = 0;
  const w = (a: Uint8Array) => { out.set(a, p); p += a.length; };
  w(outerTag); w(outerLen);
  w(f3tag); w(f3len); w(audioData);
  w(f4tag); w(f4val);
  w(f5tag); w(f5val);
  return out;
}

type DecodedFrame =
  | { type: "audio"; audio: Uint8Array; sampleRate: number; numChannels: number }
  | { type: "message"; data: Record<string, unknown> }
  | { type: "text"; text: string }
  | { type: "transcription"; text: string }
  | null;

function decodeFrame(buf: Uint8Array): DecodedFrame {
  let pos = 0;
  while (pos < buf.length) {
    const [tag, next] = decodeVarint(buf, pos);
    pos = next;
    const field = tag >>> 3;
    const wire = tag & 0x7;
    if (wire === 2) {
      const [len, start] = decodeVarint(buf, pos);
      pos = start;
      const data = buf.subarray(pos, pos + len);
      pos += len;
      if (field === 2) return decodeAudioRaw(data);
      if (field === 4) return decodeMsg(data);
      if (field === 1) return decodeTxt(data);
      if (field === 3) return decodeTranscription(data);
    } else if (wire === 0) {
      const [, np] = decodeVarint(buf, pos);
      pos = np;
    }
  }
  return null;
}

function decodeAudioRaw(buf: Uint8Array): DecodedFrame {
  let audio = new Uint8Array(0);
  let sampleRate = 24000;
  let numChannels = 1;
  let pos = 0;
  while (pos < buf.length) {
    const [tag, next] = decodeVarint(buf, pos);
    pos = next;
    const field = tag >>> 3;
    const wire = tag & 0x7;
    if (wire === 2) {
      const [len, start] = decodeVarint(buf, pos);
      pos = start;
      if (field === 3) audio = buf.subarray(pos, pos + len);
      pos += len;
    } else if (wire === 0) {
      const [val, np] = decodeVarint(buf, pos);
      pos = np;
      if (field === 4) sampleRate = val;
      else if (field === 5) numChannels = val;
    }
  }
  return { type: "audio", audio, sampleRate, numChannels };
}

function decodeMsg(buf: Uint8Array): DecodedFrame {
  let pos = 0;
  while (pos < buf.length) {
    const [tag, next] = decodeVarint(buf, pos);
    pos = next;
    const wire = tag & 0x7;
    if (wire === 2) {
      const [len, start] = decodeVarint(buf, pos);
      pos = start;
      const text = new TextDecoder().decode(buf.subarray(pos, pos + len));
      pos += len;
      try { return { type: "message", data: JSON.parse(text) }; }
      catch { return { type: "message", data: { raw: text } }; }
    } else if (wire === 0) {
      const [, np] = decodeVarint(buf, pos);
      pos = np;
    }
  }
  return null;
}

function decodeTxt(buf: Uint8Array): DecodedFrame {
  let text = "";
  let pos = 0;
  while (pos < buf.length) {
    const [tag, next] = decodeVarint(buf, pos);
    pos = next;
    const field = tag >>> 3;
    const wire = tag & 0x7;
    if (wire === 2) {
      const [len, start] = decodeVarint(buf, pos);
      pos = start;
      if (field === 3) text = new TextDecoder().decode(buf.subarray(pos, pos + len));
      pos += len;
    } else if (wire === 0) {
      const [, np] = decodeVarint(buf, pos);
      pos = np;
    }
  }
  return { type: "text", text };
}

function decodeTranscription(buf: Uint8Array): DecodedFrame {
  let text = "";
  let pos = 0;
  while (pos < buf.length) {
    const [tag, next] = decodeVarint(buf, pos);
    pos = next;
    const field = tag >>> 3;
    const wire = tag & 0x7;
    if (wire === 2) {
      const [len, start] = decodeVarint(buf, pos);
      pos = start;
      if (field === 3) text = new TextDecoder().decode(buf.subarray(pos, pos + len));
      pos += len;
    } else if (wire === 0) {
      const [, np] = decodeVarint(buf, pos);
      pos = np;
    }
  }
  return { type: "transcription", text };
}

function float32ToInt16(f32: Float32Array): Int16Array {
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return i16;
}

// ---------------------------------------------------------------------------
// Buffered PCM player — batches chunks into large blocks for smooth playback.
// Uses AudioBufferSourceNode which natively handles sample rate conversion.
// ---------------------------------------------------------------------------

class PcmPlayer {
  private ctx: AudioContext;
  private nextTime = 0;
  private chunks: Float32Array[] = [];
  private chunksLen = 0;
  private inputRate = 24000;
  private flushTimer: number | null = null;
  private activeSources: AudioBufferSourceNode[] = [];
  // Batch 500ms of audio (12000 samples @ 24kHz) before scheduling.
  private readonly MIN_BUFFER = 12000;
  // If nextTime is more than this far ahead of now, old audio from an
  // interrupted utterance is blocking — cancel it and play immediately.
  private readonly STALE_THRESHOLD_S = 1.5;

  constructor() {
    this.ctx = new AudioContext();
    // Eagerly resume — browsers may create it in "suspended" state.
    // Called within user gesture (button click) so this is allowed.
    this.ctx.resume();
  }

  play(pcm16: Uint8Array, sampleRate: number) {
    if (this.ctx.state === "suspended") this.ctx.resume();
    this.inputRate = sampleRate || 24000;

    // Copy into an aligned buffer — the subarray from protobuf decoding may
    // have an odd byteOffset which causes Int16Array to throw RangeError.
    const aligned = new Uint8Array(pcm16.length);
    aligned.set(pcm16);
    const samples = new Int16Array(aligned.buffer, 0, aligned.length / 2);
    const float32 = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) float32[i] = samples[i] / 32768;

    this.chunks.push(float32);
    this.chunksLen += float32.length;

    if (this.chunksLen >= this.MIN_BUFFER) {
      this.flush();
    } else if (!this.flushTimer) {
      // Flush after 100ms for end-of-utterance tail
      this.flushTimer = window.setTimeout(() => this.flush(), 100);
    }
  }

  /** Cancel all scheduled audio and reset timing. */
  private cancelScheduled() {
    for (const src of this.activeSources) {
      try { src.stop(); } catch { /* already stopped */ }
    }
    this.activeSources = [];
    this.nextTime = 0;
  }

  private flush() {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
    if (this.chunks.length === 0) return;

    const merged = new Float32Array(this.chunksLen);
    let offset = 0;
    for (const chunk of this.chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    this.chunks = [];
    this.chunksLen = 0;

    // createBuffer handles sample rate conversion natively (24kHz → system rate)
    const buf = this.ctx.createBuffer(1, merged.length, this.inputRate);
    buf.copyToChannel(merged, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);

    const now = this.ctx.currentTime;
    if (this.nextTime < now) {
      // Fallen behind — start with small headroom
      this.nextTime = now + 0.05;
    } else if (this.nextTime > now + this.STALE_THRESHOLD_S) {
      // nextTime is far in the future — stale audio from an interrupted
      // utterance (barge-in). Cancel old audio, play new audio immediately.
      this.cancelScheduled();
      this.nextTime = now + 0.05;
    }

    src.start(this.nextTime);
    this.nextTime += merged.length / this.inputRate;

    // Track source for cleanup; remove when playback finishes naturally
    this.activeSources.push(src);
    src.onended = () => {
      const idx = this.activeSources.indexOf(src);
      if (idx !== -1) this.activeSources.splice(idx, 1);
    };
  }

  stop() {
    if (this.flushTimer) clearTimeout(this.flushTimer);
    this.cancelScheduled();
    this.chunks = [];
    this.chunksLen = 0;
    try { this.ctx.close(); } catch { /* ignore */ }
  }
}

// ---------------------------------------------------------------------------
// Cleanup helper for voice session resources
// ---------------------------------------------------------------------------

interface VoiceResources {
  ws: WebSocket;
  micStream: MediaStream;
  micCtx: AudioContext;
  player: PcmPlayer;
}

function cleanupVoice(res: VoiceResources | null) {
  if (!res) return;
  try { res.ws.close(); } catch { /* */ }
  try { res.micStream.getTracks().forEach((t) => t.stop()); } catch { /* */ }
  try { res.micCtx.close(); } catch { /* */ }
  res.player.stop();
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function Home() {
  const [callStatus, setCallStatus] = useState<CallStatus>("idle");
  const [statusText, setStatusText] = useState("");
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [error, setError] = useState("");
  const [prospectMode, setProspectMode] = useState<ProspectMode>("human");
  const [turnLog, setTurnLog] = useState<TurnLog[]>([]);
  const voiceRef = useRef<VoiceResources | null>(null);
  const sessionIdRef = useRef<string>("");
  const abortRef = useRef(false);
  const callEndedRef = useRef(false);
  const turnLogEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll turn log when new entries arrive
  useEffect(() => {
    turnLogEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turnLog]);

  const handleCallEnded = useCallback(async () => {
    if (callEndedRef.current) return; // prevent double-calling
    callEndedRef.current = true;
    const sid = sessionIdRef.current;
    if (!sid) return;
    setCallStatus("ended");
    setStatusText("Call ended. Fetching outcome...");
    try {
      // Retry up to 3 times with increasing delay to let backend finalize
      for (let attempt = 0; attempt < 3; attempt++) {
        await new Promise((r) => setTimeout(r, 1000 + attempt * 1000));
        const res = await fetchOutcome(sid);
        if (res.outcome) {
          setOutcome(res.outcome);
          setStatusText("Completed");
          return;
        }
      }
      setStatusText("Call ended (no outcome yet)");
    } catch {
      setStatusText("Call ended (failed to fetch outcome)");
    }
  }, []);

  // --- Voice call (human prospect mode) — raw WebSocket + getUserMedia ---
  const startVoiceCall = useCallback(async () => {
    callEndedRef.current = false;
    setCallStatus("starting");
    setStatusText("Starting session...");
    setOutcome(null);
    setError("");
    setTurnLog([]);

    try {
      const run = await startSimulation("human");
      sessionIdRef.current = run.session_id;

      // Rewrite ws_url to go through Vite dev proxy (same origin)
      const wsUrl = run.connect_info.ws_url.replace(
        /^ws:\/\/[^/]+/,
        `ws://${window.location.host}`,
      );

      setCallStatus("connecting");
      setStatusText("Connecting to voice pipeline...");

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      const player = new PcmPlayer();

      // Request mic with echo cancellation + noise suppression
      const micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: { ideal: 16000 },
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Mic capture AudioContext — try 16 kHz, fall back to default
      const micCtx = new AudioContext({ sampleRate: 16000 });
      const actualRate = micCtx.sampleRate;

      const source = micCtx.createMediaStreamSource(micStream);
      // ScriptProcessorNode is deprecated in favor of AudioWorkletNode, but
      // AudioWorklet requires a separate JS file served with correct CORS
      // headers and adds deployment complexity. For this prototype,
      // ScriptProcessor is simpler and works reliably across browsers.
      const scriptNode = micCtx.createScriptProcessor(512, 1, 1);

      scriptNode.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const float32 = e.inputBuffer.getChannelData(0);
        const int16 = float32ToInt16(float32);
        const frame = encodeAudioFrame(
          int16.buffer,
          actualRate,
          1,
        );
        ws.send(frame);
      };

      source.connect(scriptNode);
      // Connect to a silent output (gain=0) so onaudioprocess fires
      // without routing mic audio to speakers (which would cause echo)
      const silentDest = micCtx.createGain();
      silentDest.gain.value = 0;
      silentDest.connect(micCtx.destination);
      scriptNode.connect(silentDest);

      voiceRef.current = { ws, micStream, micCtx, player };

      ws.onopen = () => {
        setCallStatus("in-call");
        setStatusText("In call — speak to the agent");
      };

      let audioFrameCount = 0;
      ws.onmessage = (event) => {
        try {
          const buf = new Uint8Array(event.data as ArrayBuffer);
          const parsed = decodeFrame(buf);
          if (!parsed) {
            console.log("[ws] unparsed frame, bytes:", buf.length);
            return;
          }

          if (parsed.type === "audio") {
            audioFrameCount++;
            if (audioFrameCount <= 3 || audioFrameCount % 50 === 0) {
              console.log(`[ws] audio frame #${audioFrameCount}, bytes:${parsed.audio.length}, rate:${parsed.sampleRate}`);
            }
            player.play(parsed.audio, parsed.sampleRate);
          } else if (parsed.type === "message") {
            const d = parsed.data as Record<string, unknown>;
            console.log("[ws] message:", d.type);
            if (d.type === "bot-ready") {
              setStatusText("Agent ready — listening...");
            }
          } else if (parsed.type === "text" && parsed.text.trim()) {
            console.log("[ws] agent text:", parsed.text.slice(0, 60));
            setTurnLog((prev) => [...prev, { role: "agent", text: parsed.text }]);
          } else if (parsed.type === "transcription" && parsed.text.trim()) {
            console.log("[ws] user transcription:", parsed.text.slice(0, 60));
            setTurnLog((prev) => [...prev, { role: "prospect", text: parsed.text }]);
          }
        } catch (err) {
          console.error("ws.onmessage error:", err);
        }
      };

      ws.onclose = () => {
        cleanupVoice(voiceRef.current);
        voiceRef.current = null;
        handleCallEnded();
      };

      ws.onerror = () => {
        setError("WebSocket error");
      };
    } catch (err) {
      setError(String(err));
      setCallStatus("idle");
      setStatusText("");
    }
  }, [handleCallEnded]);

  // --- AI simulation (ai prospect mode) ---
  const startAiSimulation = useCallback(async () => {
    callEndedRef.current = false;
    setCallStatus("starting");
    setStatusText("Starting AI simulation...");
    setOutcome(null);
    setError("");
    setTurnLog([]);
    abortRef.current = false;

    try {
      const run = await startSimulation("ai");
      sessionIdRef.current = run.session_id;

      setTurnLog([{ role: "agent", text: run.agent_text }]);
      setCallStatus("in-call");
      setStatusText("AI simulation in progress...");

      let ended = false;
      while (!ended && !abortRef.current) {
        await new Promise((r) => setTimeout(r, 800));

        const turn: TurnResponse = await fetchProspectTurn(run.session_id);

        setTurnLog((prev) => {
          const next = [...prev];
          if (turn.prospect_text) {
            next.push({ role: "prospect", text: turn.prospect_text });
          }
          if (turn.agent_text) {
            next.push({ role: "agent", text: turn.agent_text });
          }
          return next;
        });

        if (turn.ended) {
          ended = true;
          if (turn.outcome) {
            setOutcome(turn.outcome);
            setStatusText("Completed");
          } else {
            const res = await fetchOutcome(run.session_id);
            if (res.outcome) setOutcome(res.outcome);
            setStatusText("Completed");
          }
        }
      }

      setCallStatus("ended");
    } catch (err) {
      setError(String(err));
      setCallStatus("ended");
      setStatusText("AI simulation ended with error");
      try {
        const sid = sessionIdRef.current;
        if (sid) {
          const res = await fetchOutcome(sid);
          if (res.outcome) setOutcome(res.outcome);
        }
      } catch {
        // ignore
      }
    }
  }, []);

  const endCall = useCallback(() => {
    cleanupVoice(voiceRef.current);
    voiceRef.current = null;
    abortRef.current = true;
    handleCallEnded();
  }, [handleCallEnded]);

  const handleStart = useCallback(() => {
    if (prospectMode === "human") {
      startVoiceCall();
    } else {
      startAiSimulation();
    }
  }, [prospectMode, startVoiceCall, startAiSimulation]);

  const isActive =
    callStatus === "starting" ||
    callStatus === "connecting" ||
    callStatus === "in-call";

  return (
    <div
      style={{
        maxWidth: 720,
        margin: "2rem auto",
        fontFamily: "system-ui, sans-serif",
        padding: "0 1rem",
      }}
    >
      <h1>Roister Cold-Call Simulation</h1>
      <p style={{ color: "#666" }}>
        Start a voice cold-call (Human mode) or watch an AI prospect play
        through the conversation automatically (AI mode).
      </p>

      {/* Prospect mode toggle */}
      {!isActive && callStatus !== "ended" && (
        <div style={{ marginBottom: "1rem", display: "flex", gap: "0.5rem" }}>
          <button
            onClick={() => setProspectMode("human")}
            style={{
              padding: "0.4rem 1rem",
              borderRadius: "6px",
              border:
                prospectMode === "human"
                  ? "2px solid #7c3aed"
                  : "1px solid #ccc",
              background: prospectMode === "human" ? "#ede9fe" : "#fff",
              cursor: "pointer",
              fontWeight: prospectMode === "human" ? 600 : 400,
            }}
          >
            Human (Mic)
          </button>
          <button
            onClick={() => setProspectMode("ai")}
            style={{
              padding: "0.4rem 1rem",
              borderRadius: "6px",
              border:
                prospectMode === "ai"
                  ? "2px solid #7c3aed"
                  : "1px solid #ccc",
              background: prospectMode === "ai" ? "#ede9fe" : "#fff",
              cursor: "pointer",
              fontWeight: prospectMode === "ai" ? 600 : 400,
            }}
          >
            AI Prospect
          </button>
        </div>
      )}

      <div style={{ display: "flex", gap: "0.75rem" }}>
        <button
          onClick={handleStart}
          disabled={isActive}
          style={{
            padding: "0.6rem 1.5rem",
            fontSize: "1rem",
            cursor: isActive ? "not-allowed" : "pointer",
            borderRadius: "6px",
            border: "none",
            background: isActive ? "#888" : "#7c3aed",
            color: "#fff",
          }}
        >
          {callStatus === "idle" || callStatus === "ended"
            ? prospectMode === "human"
              ? "Start Voice Call"
              : "Run AI Simulation"
            : "Starting..."}
        </button>

        {(callStatus === "connecting" || callStatus === "in-call") && (
          <button
            onClick={endCall}
            style={{
              padding: "0.6rem 1.5rem",
              fontSize: "1rem",
              cursor: "pointer",
              borderRadius: "6px",
              border: "none",
              background: "#dc2626",
              color: "#fff",
            }}
          >
            {prospectMode === "human" ? "End Call" : "Stop"}
          </button>
        )}
      </div>

      {statusText && (
        <p style={{ marginTop: "1rem", fontStyle: "italic" }}>{statusText}</p>
      )}

      {error && (
        <p style={{ marginTop: "0.5rem", color: "#dc2626" }}>{error}</p>
      )}

      {callStatus === "in-call" && prospectMode === "human" && (
        <div
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            border: "2px solid #7c3aed",
            borderRadius: "8px",
            textAlign: "center",
          }}
        >
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: "50%",
              background: "#22c55e",
              display: "inline-block",
              marginRight: 8,
              animation: "pulse 1.5s infinite",
            }}
          />
          <span>Live — microphone active (barge-in enabled)</span>
        </div>
      )}

      {/* Turn log for AI mode */}
      {turnLog.length > 0 && (
        <div
          style={{
            marginTop: "1.5rem",
            border: "1px solid #e2e8f0",
            borderRadius: "8px",
            maxHeight: 400,
            overflow: "auto",
          }}
        >
          {turnLog.map((t, i) => (
            <div
              key={i}
              style={{
                padding: "0.5rem 0.75rem",
                borderBottom: "1px solid #f1f5f9",
                background: t.role === "agent" ? "#f8fafc" : "#fff",
              }}
            >
              <span
                style={{
                  fontWeight: 600,
                  color: t.role === "agent" ? "#7c3aed" : "#2563eb",
                  fontSize: "0.8rem",
                  textTransform: "uppercase",
                }}
              >
                {t.role === "agent" ? "Agent" : "Prospect"}:
              </span>{" "}
              <span style={{ fontSize: "0.9rem" }}>{t.text}</span>
            </div>
          ))}
          <div ref={turnLogEndRef} />
        </div>
      )}

      {outcome && <OutcomeView outcome={outcome} />}

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
