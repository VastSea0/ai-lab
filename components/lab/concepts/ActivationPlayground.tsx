"use client";

import { line, scaleLinear } from "d3";
import { useMemo, useState } from "react";
import type { ActivationName } from "@/lib/ml/network";
import { activationDerivative, activationValue, formatNumber } from "@/lib/ml/network";

const activationOptions: ActivationName[] = ["linear", "sigmoid", "tanh", "relu"];

export function ActivationPlayground({
  initialActivation = "sigmoid",
}: {
  initialActivation?: ActivationName;
}) {
  const [activation, setActivation] = useState<ActivationName>(initialActivation);
  const [z, setZ] = useState(0);
  const value = activationValue(activation, z);
  const derivative = activationDerivative(activation, z);
  const width = 360;
  const height = 150;
  const xScale = scaleLinear().domain([-4, 4]).range([24, width - 16]);
  const yDomain = activation === "tanh" || activation === "linear" ? [-1.4, 1.4] : [-0.25, 1.25];
  const yScale = scaleLinear().domain(yDomain).range([height - 22, 12]);
  const curve = useMemo(
    () =>
      Array.from({ length: 120 }, (_, index) => {
        const pointZ = -4 + (index / 119) * 8;
        return { z: pointZ, value: activationValue(activation, pointZ) };
      }),
    [activation]
  );
  const path =
    line<{ z: number; value: number }>()
      .x((point) => xScale(point.z))
      .y((point) => yScale(point.value))(curve) ?? "";

  return (
    <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            Activation Playground
          </div>
          <div className="mt-1 text-xs text-[#526070]">z değerini oynat, çıktı ve türev nasıl değişiyor gör.</div>
        </div>
        <select
          className="h-8 rounded-md border border-[#cbd5e1] bg-white px-2 text-xs font-semibold"
          value={activation}
          onChange={(event) => setActivation(event.target.value as ActivationName)}
        >
          {activationOptions.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </div>

      <svg className="mt-3 h-[150px] w-full" viewBox={`0 0 ${width} ${height}`}>
        <rect width={width} height={height} rx={7} fill="#ffffff" stroke="#dbe3ee" />
        <line x1={24} x2={width - 16} y1={yScale(0)} y2={yScale(0)} stroke="#e2e8f0" />
        <line x1={xScale(0)} x2={xScale(0)} y1={12} y2={height - 22} stroke="#e2e8f0" />
        <path d={path} fill="none" stroke="#2563eb" strokeWidth={2.5} />
        <circle cx={xScale(z)} cy={yScale(value)} r={5} fill="#f97316" stroke="#ffffff" strokeWidth={1.5} />
        <text x={xScale(z) + 8} y={Math.max(18, yScale(value) - 8)} className="fill-[#334155] text-[10px] font-semibold">
          f(z)
        </text>
      </svg>

      <input
        className="mt-3 w-full accent-[#2563eb]"
        min={-4}
        max={4}
        step={0.05}
        type="range"
        value={z}
        onChange={(event) => setZ(Number(event.target.value))}
        aria-label="z değeri"
      />
      <div className="mt-2 grid grid-cols-3 gap-2">
        <Metric label="z" value={formatNumber(z, 3)} />
        <Metric label="f(z)" value={formatNumber(value, 5)} />
        <Metric label="f'(z)" value={formatNumber(derivative, 5)} />
      </div>
      <div className="mt-2 rounded-md border border-[#dbe3ee] bg-white p-2 text-xs leading-5 text-[#526070]">
        {derivative < 0.05
          ? "Türev çok küçük: bu noktada gradient zor akar, öğrenme yavaşlayabilir."
          : derivative > 0.9
            ? "Türev güçlü: küçük değişiklikler çıktıyı belirgin etkiler."
            : "Türev orta seviyede: gradient akışı okunabilir durumda."}
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white px-2 py-1.5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">{label}</div>
      <div className="mt-1 font-mono text-xs font-semibold text-[#18202f]">{value}</div>
    </div>
  );
}
