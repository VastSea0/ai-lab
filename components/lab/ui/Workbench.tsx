"use client";

import { ChevronDown, X } from "lucide-react";
import { useState, type ReactNode } from "react";

export interface WorkbenchTabItem<T extends string = string> {
  id: T;
  label: string;
  icon?: ReactNode;
  badge?: string;
}

export interface MetricStripItem {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "warning";
}

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function WorkbenchShell({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <main className={cx("h-screen min-h-[720px] w-screen max-w-full overflow-hidden bg-[#f4f7fb] text-[#172033]", className)}>
      {children}
    </main>
  );
}

export function MetricStrip({
  items,
  className = "",
}: {
  items: MetricStripItem[];
  className?: string;
}) {
  return (
    <div className={cx("flex min-w-0 items-center gap-1.5", className)}>
      {items.map((item) => (
        <div
          key={item.label}
          className={cx(
            "min-w-[76px] rounded-md border bg-white px-2.5 py-1.5",
            item.tone === "good"
              ? "border-[#bbf7d0] text-[#047857]"
              : item.tone === "warning"
                ? "border-[#fed7aa] text-[#b45309]"
                : "border-[#dbe3ee] text-[#172033]"
          )}
        >
          <div className="truncate text-[9px] font-semibold uppercase tracking-[0.08em] text-[#69778d]">
            {item.label}
          </div>
          <div className="truncate text-[13px] font-semibold leading-5" title={item.value}>
            {item.value}
          </div>
        </div>
      ))}
    </div>
  );
}

export function PanelTabs<T extends string>({
  items,
  value,
  onChange,
  className = "",
  compact = false,
}: {
  items: Array<WorkbenchTabItem<T>>;
  value: T;
  onChange: (value: T) => void;
  className?: string;
  compact?: boolean;
}) {
  return (
    <div
      className={cx("grid rounded-md border border-[#dbe3ee] bg-[#f8fafc] p-1", className)}
      style={{ gridTemplateColumns: `repeat(${items.length}, minmax(0, 1fr))` }}
    >
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          className={cx(
            "inline-flex min-w-0 items-center justify-center gap-1.5 rounded-[5px] px-2 font-semibold transition",
            compact ? "h-7 text-[11px]" : "h-8 text-xs",
            value === item.id
              ? "bg-white text-[#172033] shadow-sm"
              : "text-[#64748b] hover:bg-white/70 hover:text-[#2563eb]"
          )}
          onClick={() => onChange(item.id)}
        >
          {item.icon && <span className="shrink-0">{item.icon}</span>}
          <span className="truncate">{item.label}</span>
          {item.badge && (
            <span className="rounded bg-[#e8f0ff] px-1.5 py-0.5 text-[10px] text-[#2563eb]">
              {item.badge}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

export function IconToolbar({
  children,
  className = "",
  label,
}: {
  children: ReactNode;
  className?: string;
  label?: string;
}) {
  return (
    <div
      className={cx(
        "inline-flex items-center gap-1 rounded-md border border-[#d5deea] bg-white/92 p-1 shadow-sm backdrop-blur",
        className
      )}
      aria-label={label}
    >
      {children}
    </div>
  );
}

export function CompactPanel({
  title,
  icon,
  actions,
  children,
  className = "",
  bodyClassName = "",
}: {
  title?: string;
  icon?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cx("rounded-md border border-[#dbe3ee] bg-white", className)}>
      {(title || actions) && (
        <div className="flex min-h-10 items-center justify-between gap-3 border-b border-[#edf2f7] px-3 py-2">
          <div className="flex min-w-0 items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            {icon && <span className="shrink-0 text-[#2563eb]">{icon}</span>}
            {title && <span className="truncate">{title}</span>}
          </div>
          {actions && <div className="shrink-0">{actions}</div>}
        </div>
      )}
      <div className={cx("p-3", bodyClassName)}>{children}</div>
    </section>
  );
}

export function DrawerPanel({
  title,
  icon,
  summary,
  children,
  defaultOpen = false,
  className = "",
}: {
  title: string;
  icon?: ReactNode;
  summary?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <details
      className={cx("group rounded-md border border-[#dbe3ee] bg-white", className)}
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2">
        <span className="flex min-w-0 items-center gap-2">
          {icon && <span className="shrink-0 text-[#2563eb]">{icon}</span>}
          <span className="truncate text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            {title}
          </span>
          {summary && <span className="truncate text-xs text-[#526070]">{summary}</span>}
        </span>
        <ChevronDown className="h-4 w-4 shrink-0 text-[#64748b] transition group-open:rotate-180" />
      </summary>
      <div className="border-t border-[#edf2f7] p-3">{children}</div>
    </details>
  );
}

export function StatusToast({
  title,
  children,
  icon,
  action,
  onClose,
  className = "",
}: {
  title: string;
  children?: ReactNode;
  icon?: ReactNode;
  action?: ReactNode;
  onClose?: () => void;
  className?: string;
}) {
  return (
    <div
      className={cx(
        "pointer-events-auto rounded-md border border-white/70 bg-white/[0.92] px-3 py-2 text-xs text-[#526070] shadow-sm backdrop-blur",
        className
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-2 font-semibold text-[#172033]">
          {icon && <span className="shrink-0 text-[#2563eb]">{icon}</span>}
          <span className="truncate">{title}</span>
        </div>
        {onClose && (
          <button
            type="button"
            className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded border border-[#dbe3ee] text-[#64748b] hover:border-[#ef4444] hover:text-[#b91c1c]"
            onClick={onClose}
            aria-label="Kapat"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
      {children && <div className="mt-1 leading-5">{children}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}
