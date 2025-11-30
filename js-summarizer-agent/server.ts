
// server.ts
import "dotenv/config"; // Load environment variables
import express from "express";
import { InMemoryTaskStore, DefaultRequestHandler } from "@a2a-js/sdk/server";
import { A2AExpressApp } from "@a2a-js/sdk/server/express";

// import your executor and agent card
import { SummarizerAgentExecutor } from "./summarizerAgentExecutor.js";
import { summarizerAgentCard } from "./agentCard.js";

const PORT = parseInt(process.env.PORT || "4100", 10);

// 1) Task store and executor
const taskStore = new InMemoryTaskStore();
const executor = new SummarizerAgentExecutor();

// 2) Standard A2A request handler
const requestHandler = new DefaultRequestHandler(summarizerAgentCard, taskStore, executor);

// 3) Mount A2A protocol endpoints onto Express
const appBuilder = new A2AExpressApp(requestHandler);
const app = appBuilder.setupRoutes(express()); // auto-adds /.well-known/agent.json + POST /

app.listen(PORT, () => {
  console.log(`JS A2A agent listening on http://localhost:${PORT}`);
  console.log(`AgentCard: http://localhost:${PORT}/.well-known/agent.json`);
});
