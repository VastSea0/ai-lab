import { SandboxApp } from "@/components/SandboxApp";

interface HomeProps {
  searchParams?: Promise<{
    view?: string;
  }>;
}

export default async function Home({ searchParams }: HomeProps) {
  const params = await searchParams;
  return <SandboxApp initialView={params?.view === "3d" ? "surface3d" : "network"} />;
}
