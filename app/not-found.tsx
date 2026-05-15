import Link from "next/link";

export default function NotFound() {
  return (
    <main className="grid min-h-screen place-items-center bg-[#f5f7fb] px-6 text-[#18202f]">
      <div className="max-w-md rounded-md border border-[#dbe3ee] bg-white p-6 text-center shadow-panel">
        <h1 className="text-xl font-semibold">Sayfa bulunamadı</h1>
        <p className="mt-2 text-sm text-[#64748b]">Sandbox ana ekranına dönebilirsin.</p>
        <Link
          className="mt-5 inline-flex h-9 items-center rounded-md border border-[#cbd5e1] px-4 text-sm font-semibold text-[#2563eb]"
          href="/"
        >
          Ana ekran
        </Link>
      </div>
    </main>
  );
}
