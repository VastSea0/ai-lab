import type { Metadata } from "next";
import { AiCodeLabWorkspace } from "@/components/lab/code/AiCodeLabWorkspace";

export const metadata: Metadata = {
  title: "AI Kod Labi",
  description: "Python ile ML, deep learning ve reinforcement challenge calisma alani",
};

export default function CodeLabPage() {
  return <AiCodeLabWorkspace />;
}
