
// agentCard.ts
import type { AgentCard } from "@a2a-js/sdk";

export const summarizerAgentCard: AgentCard = {
  name: "InterAgentX JS Summarizer",
  description:
    "Final summarizer that aggregates outputs from Python RAG Analyst, Web Researcher, and Evidence Verifier; streams concise summary with citations.",
  protocolVersion: "0.3.0",
  version: "0.1.0",
  url: "http://localhost:4100/",
  defaultInputModes: ["text"],
  defaultOutputModes: ["text"],
  capabilities: {
    streaming: true,
  },
  skills: [
    {
      id: "summary",
      name: "Cross-agent summarization",
      description: "Summarize prior agent outputs and preserve citations.",
      tags: ["summary", "citations", "streaming"],
    },
  ],
};
