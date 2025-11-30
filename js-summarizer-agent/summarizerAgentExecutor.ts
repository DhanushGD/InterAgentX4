// summarizerAgentExecutor.ts
import type { AgentExecutor, RequestContext, ExecutionEventBus } from "@a2a-js/sdk/server";
import type { Message } from "@a2a-js/sdk";
import { v4 as uuidv4 } from "uuid";
import OpenAI from "openai";

export class SummarizerAgentExecutor implements AgentExecutor {
  private openai: OpenAI;

  constructor() {
    // Initialize OpenAI client
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey || apiKey === 'your_openai_api_key_here') {
      console.warn("OPENAI_API_KEY not set or is placeholder. OpenAI features will be disabled.");
      this.openai = null as any;
    } else {
      this.openai = new OpenAI({ apiKey });
    }
  }

  async execute(
    requestContext: RequestContext,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    try {
      console.log("[JS SUMMARIZER] Received request for summarization");

      // Extract user input from the request
      const userMessage = requestContext.userMessage;
      let userText = "";

      if (userMessage && userMessage.parts) {
        // Extract text from message parts
        for (const part of userMessage.parts) {
          if (part.kind === "text") {
            userText += part.text + " ";
          }
        }
      }

      userText = userText.trim();
      console.log(`[JS SUMMARIZER] Processing input: ${userText.substring(0, 100)}...`);

      if (!userText) {
        throw new Error("No input provided for summarization.");
      }

      // Try to parse structured data from RAG Analyst
      let structuredData: any = null;
      try {
        // Look for JSON data in the message
        const jsonMatch = userText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          structuredData = JSON.parse(jsonMatch[0]);
        }
      } catch (e) {
        console.log("Could not parse structured data, treating as plain text");
      }

      let summary: string;

      if (structuredData) {
        console.log("[JS SUMMARIZER] Creating structured summary from agent outputs");
        // Use OpenAI to create a comprehensive summary from structured agent outputs
        summary = await this.createStructuredSummary(structuredData);
      } else {
        console.log("[JS SUMMARIZER] Creating text summary");
        // Fallback for plain text summarization
        summary = await this.createTextSummary(userText);
      }

      console.log(`[JS SUMMARIZER] Generated summary: ${summary.substring(0, 100)}...`);

      // Create response message
      const responseMessage: Message = {
        kind: "message",
        messageId: uuidv4(),
        role: "agent",
        parts: [{ kind: "text", text: summary }],
        contextId: requestContext.contextId,
      };

      console.log("[JS SUMMARIZER] Publishing response");
      // Publish the message
      eventBus.publish(responseMessage);

      // Signal completion
      eventBus.finished();
      console.log("[JS SUMMARIZER] Request completed successfully");

    } catch (error) {
      console.error("Error in SummarizerAgentExecutor:", error);

      // Send error message
      const errorMessage: Message = {
        kind: "message",
        messageId: uuidv4(),
        role: "agent",
        parts: [{ kind: "text", text: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}` }],
        contextId: requestContext.contextId,
      };

      eventBus.publish(errorMessage);
      eventBus.finished();
    }
  }

  private async createStructuredSummary(data: any): Promise<string> {
    if (!this.openai) {
      // Fallback when OpenAI is not available
      return this.createFallbackStructuredSummary(data);
    }

    const prompt = `You are an expert summarizer for a multi-agent analysis system.

Based on the following structured analysis results from different AI agents, create a comprehensive, well-structured summary that synthesizes all the information with proper citations and maintains academic rigor.

**Original Query:** ${data.query || 'Not specified'}

**Local Context Analysis (RAG Draft):**
${data.rag_draft || 'Not available'}

**Web Research Evidence:**
${data.web_evidence || 'Not available'}

**Evidence Verification:**
${data.evidence_verification || 'Not available'}

**Crew AI Synthesis:**
${data.crew_synthesis || 'Not available'}

Please provide a summary that:
1. Starts with a clear, concise answer to the query
2. Integrates findings from all agents
3. Highlights key evidence and citations
4. Notes any conflicts or uncertainties
5. Maintains objectivity and academic tone
6. Is well-structured and easy to read

Format your response professionally with clear sections and proper citations.`;

    const completion = await this.openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || "gpt-4-turbo",
      messages: [
        {
          role: "system",
          content: "You are a professional summarizer that synthesizes multi-agent analysis results into coherent, well-cited summaries."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.3,
      max_tokens: 2000,
    });

    return completion.choices[0]?.message?.content || "Error: No summary generated";
  }

  private createFallbackStructuredSummary(data: any): string {
    return `## InterAgentX Analysis Summary

**Query:** ${data.query || 'Not specified'}

### 📚 Local Context Analysis (RAG Draft)
${data.rag_draft || 'Analysis not available - RAG system may not have relevant local context.'}

### 🔍 Web Research Evidence
${data.web_evidence || 'Web research not performed or no external evidence found.'}

### ✅ Evidence Verification
${data.evidence_verification || 'Evidence verification not completed.'}

### 🤖 Crew AI Synthesis
${data.crew_synthesis || 'AI synthesis not available.'}

**Note:** This is a fallback summary generated without OpenAI API. Please configure OPENAI_API_KEY for enhanced summarization capabilities.`;
  }

  private async createTextSummary(text: string): Promise<string> {
    if (!this.openai) {
      // Fallback when OpenAI is not available
      return `## Summary

${text.substring(0, 500)}${text.length > 500 ? '...' : ''}

**Note:** This is a basic text extraction summary. Configure OPENAI_API_KEY for AI-powered summarization.`;
    }

    const prompt = `Please provide a comprehensive summary of the following text. Focus on the main points, key findings, and any important conclusions:

${text}

Summary:`;

    const completion = await this.openai.chat.completions.create({
      model: process.env.OPENAI_MODEL || "gpt-4-turbo",
      messages: [
        {
          role: "system",
          content: "You are a professional summarizer that creates clear, concise summaries of complex information."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.3,
      max_tokens: 1000,
    });

    return completion.choices[0]?.message?.content || "Error: No summary generated";
  }

  async cancelTask(taskId: string, eventBus: ExecutionEventBus): Promise<void> {
    // Implement cancellation logic if needed
    eventBus.finished();
  }
}
