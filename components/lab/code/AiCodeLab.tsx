"use client";

import {
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  Code2,
  Cpu,
  FlaskConical,
  Gamepad2,
  Layers3,
  Play,
  RotateCcw,
  UploadCloud,
} from "lucide-react";
import { useMemo, useState } from "react";
import {
  AI_CODE_LAB_CHALLENGES,
  type AiCodeLabChallenge,
  type CodeLabTrack,
  type PythonLabResult,
  type PythonLabRunResponse,
} from "@/lib/ml/code-lab";
import { formatNumber } from "@/lib/ml/network";

const trackMeta: Record<CodeLabTrack, { label: string; icon: React.ReactNode }> = {
  "core-ml": { label: "ML", icon: <FlaskConical className="h-3.5 w-3.5" /> },
  "deep-learning": { label: "Deep", icon: <BrainCircuit className="h-3.5 w-3.5" /> },
  reinforcement: { label: "RL", icon: <Gamepad2 className="h-3.5 w-3.5" /> },
};

const difficultyClass: Record<AiCodeLabChallenge["difficulty"], string> = {
  kolay: "border-[#10b981] bg-[#ecfdf5] text-[#047857]",
  orta: "border-[#f59e0b] bg-[#fffbeb] text-[#b45309]",
  zor: "border-[#ef4444] bg-[#fef2f2] text-[#b91c1c]",
};

export function AiCodeLab({
  onApplyResult,
  applyLabel = "Ağa Aktar",
}: {
  onApplyResult: (challenge: AiCodeLabChallenge, result: PythonLabResult) => void;
  applyLabel?: string;
}) {
  const [track, setTrack] = useState<CodeLabTrack>("core-ml");
  const visibleChallenges = useMemo(
    () => AI_CODE_LAB_CHALLENGES.filter((challenge) => challenge.track === track),
    [track]
  );
  const [selectedId, setSelectedId] = useState(AI_CODE_LAB_CHALLENGES[0].id);
  const selected =
    AI_CODE_LAB_CHALLENGES.find((challenge) => challenge.id === selectedId) ?? AI_CODE_LAB_CHALLENGES[0];
  const [code, setCode] = useState(AI_CODE_LAB_CHALLENGES[0].starterCode);
  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<PythonLabRunResponse | null>(null);

  const selectChallenge = (challenge: AiCodeLabChallenge) => {
    setSelectedId(challenge.id);
    setCode(challenge.starterCode);
    setResponse(null);
  };

  const selectTrack = (nextTrack: CodeLabTrack) => {
    setTrack(nextTrack);
    selectChallenge(
      AI_CODE_LAB_CHALLENGES.find((challenge) => challenge.track === nextTrack) ?? AI_CODE_LAB_CHALLENGES[0]
    );
  };

  const runCode = async () => {
    setRunning(true);
    setResponse(null);
    try {
      const result = await fetch("/api/python-lab", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ challengeId: selected.id, code }),
      });
      const payload = (await result.json()) as PythonLabRunResponse;
      setResponse(payload);
    } catch (error) {
      setResponse({
        ok: false,
        stdout: "",
        stderr: "",
        durationMs: 0,
        error: error instanceof Error ? error.message : "Python hücresi çalıştırılamadı.",
      });
    } finally {
      setRunning(false);
    }
  };

  const canApply =
    selected.applyToNetwork &&
    Boolean(response?.result?.layers?.length && response.result.weights?.length && response.result.biases?.length);

  return (
    <div className="space-y-3">
      <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
        <div className="flex items-center gap-2 text-sm font-semibold text-[#18202f]">
          <Code2 className="h-4 w-4 text-[#2563eb]" />
          LeetCode AI Kod Labı
        </div>
        <div className="mt-2 grid grid-cols-3 gap-1 rounded-md border border-[#dbe3ee] bg-white p-1">
          {(Object.keys(trackMeta) as CodeLabTrack[]).map((item) => (
            <button
              key={item}
              type="button"
              className={`inline-flex h-8 items-center justify-center gap-1 rounded px-2 text-[11px] font-semibold ${
                track === item ? "bg-[#2563eb] text-white" : "text-[#64748b] hover:bg-[#eef4ff] hover:text-[#2563eb]"
              }`}
              onClick={() => selectTrack(item)}
            >
              {trackMeta[item].icon}
              {trackMeta[item].label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        {visibleChallenges.map((challenge) => (
          <button
            key={challenge.id}
            type="button"
            className={`block w-full rounded-md border p-3 text-left transition ${
              selected.id === challenge.id
                ? "border-[#2563eb] bg-[#eef4ff]"
                : "border-[#dbe3ee] bg-white hover:border-[#2563eb]"
            }`}
            onClick={() => selectChallenge(challenge)}
          >
            <div className="flex items-start justify-between gap-2">
              <span className="min-w-0 text-xs font-semibold text-[#18202f]">{challenge.title}</span>
              <span className={`rounded border px-1.5 py-0.5 text-[10px] font-bold ${difficultyClass[challenge.difficulty]}`}>
                {challenge.difficulty}
              </span>
            </div>
            <div className="mt-1 text-[11px] leading-4 text-[#526070]">{challenge.summary}</div>
            <div className="mt-2 flex flex-wrap gap-1">
              {challenge.libraries.map((library) => (
                <span key={library} className="rounded border border-[#dbe3ee] bg-[#fbfdff] px-1.5 py-0.5 text-[10px] font-semibold text-[#64748b]">
                  {library}
                </span>
              ))}
            </div>
          </button>
        ))}
      </div>

      <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="text-xs font-semibold text-[#18202f]">{selected.title}</div>
            <div className="mt-1 text-[11px] leading-5 text-[#526070]">{selected.prompt}</div>
          </div>
          <Cpu className="h-4 w-4 shrink-0 text-[#2563eb]" />
        </div>
        <div className="mt-2 flex flex-wrap gap-1">
          {selected.concepts.map((concept) => (
            <span key={concept} className="rounded-md bg-[#f1f5f9] px-2 py-1 text-[10px] font-semibold text-[#475569]">
              {concept}
            </span>
          ))}
        </div>
      </div>

      <div className="rounded-md border border-[#dbe3ee] bg-white">
        <div className="flex items-center justify-between border-b border-[#e2e8f0] px-3 py-2">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            <Code2 className="h-4 w-4 text-[#2563eb]" />
            Python
          </div>
          <button
            type="button"
            className="inline-flex h-7 items-center gap-1 rounded border border-[#cbd5e1] bg-white px-2 text-[11px] font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
            onClick={() => setCode(selected.starterCode)}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Sıfırla
          </button>
        </div>
        <textarea
          className="h-64 w-full resize-y border-0 bg-[#0f172a] p-3 font-mono text-[11px] leading-5 text-[#e2e8f0] outline-none"
          value={code}
          onChange={(event) => setCode(event.target.value)}
          spellCheck={false}
          aria-label="Python challenge kodu"
        />
        <div className="grid grid-cols-[1fr_auto] gap-2 border-t border-[#e2e8f0] p-3">
          <button
            type="button"
            className="inline-flex h-9 items-center justify-center gap-2 rounded-md bg-[#2563eb] px-3 text-xs font-semibold text-white hover:bg-[#1d4ed8] disabled:cursor-wait disabled:opacity-60"
            disabled={running}
            onClick={runCode}
          >
            <Play className="h-4 w-4" />
            {running ? "Çalışıyor" : "Çalıştır"}
          </button>
          <button
            type="button"
            className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-[#cbd5e1] bg-white px-3 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb] disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!canApply || !response?.result}
            onClick={() => {
              if (response?.result) onApplyResult(selected, response.result);
            }}
          >
            <UploadCloud className="h-4 w-4" />
            {applyLabel}
          </button>
        </div>
      </div>

      {response && <RunResult response={response} challenge={selected} />}
    </div>
  );
}

function RunResult({
  response,
  challenge,
}: {
  response: PythonLabRunResponse;
  challenge: AiCodeLabChallenge;
}) {
  const result = response.result;
  return (
    <div className="space-y-3">
      <div
        className={`rounded-md border p-3 ${
          response.ok ? "border-[#bbf7d0] bg-[#f0fdf4]" : "border-[#fecaca] bg-[#fef2f2]"
        }`}
      >
        <div className="flex items-center gap-2 text-xs font-semibold text-[#18202f]">
          {response.ok ? (
            <CheckCircle2 className="h-4 w-4 text-[#047857]" />
          ) : (
            <AlertTriangle className="h-4 w-4 text-[#b91c1c]" />
          )}
          {response.ok ? "Python sonucu hazır" : "Python sonucu hata verdi"}
        </div>
        <div className="mt-1 text-[11px] leading-5 text-[#526070]">
          {formatNumber(response.durationMs / 1000, 2)} sn · {challenge.applyToNetwork ? "network aktarımı destekli" : "simülasyon sonucu"}
          {response.error ? ` · ${response.error}` : ""}
        </div>
      </div>

      {result && (
        <>
          <ResultMetrics result={result} />
          {result.losses && result.losses.length > 0 && <LossSparkline values={result.losses} />}
          {result.layers && <LayerPreview layers={result.layers} />}
          {result.policy && <PolicyPreview policy={result.policy} trajectory={result.trajectory ?? []} />}
          {result.notes && result.notes.length > 0 && (
            <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                Notlar
              </div>
              <div className="space-y-1 text-[11px] leading-5 text-[#526070]">
                {result.notes.map((note) => (
                  <div key={note}>{note}</div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {(response.stdout || response.stderr) && (
        <details className="rounded-md border border-[#dbe3ee] bg-white">
          <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-[#2563eb]">
            stdout / stderr
          </summary>
          <pre className="max-h-44 overflow-auto border-t border-[#e2e8f0] bg-[#0f172a] p-3 text-[10px] leading-4 text-[#e2e8f0]">
            {response.stdout}
            {response.stderr ? `\n--- stderr ---\n${response.stderr}` : ""}
          </pre>
        </details>
      )}
    </div>
  );
}

function ResultMetrics({ result }: { result: PythonLabResult }) {
  const metrics = [
    result.epochs !== undefined ? { label: "epoch", value: result.epochs } : null,
    result.accuracy !== undefined ? { label: "accuracy", value: `${formatNumber(result.accuracy * 100, 1)}%` } : null,
    result.losses?.length ? { label: "son loss", value: formatNumber(result.losses.at(-1) ?? 0, 6) } : null,
    ...(result.metrics ?? []),
  ].filter(Boolean) as Array<{ label: string; value: string | number }>;
  if (metrics.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-2">
      {metrics.slice(0, 6).map((metric) => (
        <div key={metric.label} className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-3 py-2">
          <div className="truncate text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            {metric.label}
          </div>
          <div className="mt-1 truncate text-sm font-semibold text-[#18202f]">{metric.value}</div>
        </div>
      ))}
    </div>
  );
}

function LossSparkline({ values }: { values: number[] }) {
  const width = 318;
  const height = 84;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(1e-9, max - min);
  const points = values
    .map((value, index) => {
      const x = 10 + (index / Math.max(1, values.length - 1)) * (width - 20);
      const y = height - 14 - ((value - min) / spread) * (height - 26);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        Python Loss
      </div>
      <svg className="h-[84px] w-full" viewBox={`0 0 ${width} ${height}`}>
        <rect width={width} height={height} rx={6} fill="#fbfdff" stroke="#dbe3ee" />
        <polyline points={points} fill="none" stroke="#2563eb" strokeWidth={2.6} strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={width - 10} cy={height - 14 - ((values.at(-1) ?? min) - min) / spread * (height - 26)} r={3.6} fill="#f97316" />
      </svg>
    </div>
  );
}

function LayerPreview({ layers }: { layers: PythonLabResult["layers"] }) {
  if (!layers?.length) return null;
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        <Layers3 className="h-4 w-4 text-[#2563eb]" />
        Kodlanan Nöronlar
      </div>
      <div className="flex items-center gap-3 overflow-x-auto pb-1">
        {layers.map((layer, layerIndex) => {
          const visible = Math.min(14, layer.size);
          return (
            <div key={`${layerIndex}-${layer.size}`} className="shrink-0">
              <div className="mb-1 text-center text-[10px] font-semibold text-[#64748b]">
                L{layerIndex} · {layer.size}
              </div>
              <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${Math.min(4, visible)}, 10px)` }}>
                {Array.from({ length: visible }, (_, index) => (
                  <span
                    key={index}
                    className={`block h-2.5 w-2.5 rounded-full ${
                      layerIndex === 0 ? "bg-[#38bdf8]" : layerIndex === layers.length - 1 ? "bg-[#f87171]" : "bg-[#fbbf24]"
                    }`}
                  />
                ))}
              </div>
              <div className="mt-1 text-center text-[10px] text-[#64748b]">{layer.activation ?? "linear"}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PolicyPreview({
  policy,
  trajectory,
}: {
  policy: string[];
  trajectory: number[][];
}) {
  const size = Math.round(Math.sqrt(policy.length));
  if (size <= 0 || size * size !== policy.length) return null;
  const arrows: Record<string, string> = {
    up: "↑",
    right: "→",
    down: "↓",
    left: "←",
    goal: "G",
    trap: "!",
  };
  const pathCells = new Set(trajectory.map(([x, y]) => `${x}:${y}`));
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        Policy Simülasyonu
      </div>
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}>
        {policy.map((action, index) => {
          const x = index % size;
          const y = Math.floor(index / size);
          const isPath = pathCells.has(`${x}:${y}`);
          return (
            <div
              key={`${action}-${index}`}
              className={`flex aspect-square items-center justify-center rounded-md border text-sm font-bold ${
                action === "goal"
                  ? "border-[#10b981] bg-[#ecfdf5] text-[#047857]"
                  : action === "trap"
                    ? "border-[#ef4444] bg-[#fef2f2] text-[#b91c1c]"
                    : isPath
                      ? "border-[#2563eb] bg-[#eef4ff] text-[#1d4ed8]"
                      : "border-[#dbe3ee] bg-[#fbfdff] text-[#64748b]"
              }`}
            >
              {arrows[action] ?? action.slice(0, 1)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
