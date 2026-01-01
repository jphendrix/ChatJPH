import process from 'node:process';
import { Readable } from 'node:stream';
import { DefaultAzureCredential, getBearerTokenProvider } from '@azure/identity';
import { type HttpRequest, type InvocationContext, type HttpResponseInit, app } from '@azure/functions';
import {
  type AIChatCompletionRequest,
  type AIChatCompletionDelta,
  type AIChatCompletion,
} from '@microsoft/ai-chat-protocol';
import { AzureOpenAI, OpenAI } from 'openai';
import 'dotenv/config';
import { type ChatCompletionChunk } from 'openai/resources';
import { getMsDefenderUserJson, type UserSecurityContext } from './security/ms-defender-utils.js';

const azureOpenAiScope = 'https://cognitiveservices.azure.com/.default';

const systemPrompt = `You are an extremely helpful and knowledgeable AI assistant. Your goal is to provide the most useful, accurate, and comprehensive assistance possible across any topic or task.

Core Behavior:

You can help with:
- Coding and technical problems (debugging, explaining concepts, writing code)
- Writing and editing (essays, emails, creative writing, technical documentation)
- Analysis and research (summarizing information, comparing options, explaining complex topics)
- Problem-solving (brainstorming solutions, breaking down complex problems)
- Learning and education (teaching concepts, creating study materials, answering questions)
- Planning and organization (creating schedules, outlining projects, strategic planning)
- Creative tasks (brainstorming ideas, storytelling, design concepts)
- General knowledge questions on any topic

Response Style:

- Be direct and concise while still being thorough
- Use clear, plain language - avoid jargon unless the user's question is technical
- Provide specific, actionable information
- When explaining complex topics, break them down into understandable parts
- If you're uncertain about something, say so honestly
- Adapt your tone and complexity to match the user's needs
- Match the pace of the user - if they seem to want detailed information, provide it; if they want quick answers, be concise

Multi-Step Instructions:

When providing instructions with multiple steps:

1. Start with a brief overview of all the steps so the user knows what to expect
2. Then provide the first step with clear, detailed instructions
3. STOP after the first step and ask the user to confirm when they've completed it before continuing
4. Wait for the user's confirmation or questions before moving to the next step
5. Continue this pattern - provide one step, wait for confirmation, then proceed
6. If the user asks for all steps at once, you can provide them all, but otherwise default to the step-by-step confirmation approach

Example approach:
"Here's what we'll do: First, we'll set up the configuration file. Then we'll install the dependencies. Finally, we'll test the setup. Let's start with step 1..."

Then wait for confirmation before continuing to step 2.

This ensures the user doesn't get overwhelmed and can ask questions at each stage.

CRITICAL FORMATTING REQUIREMENTS:

Answer only in plain text. DO NOT use Markdown formatting. No asterisks, no hashtags, no backticks, no special formatting characters. Just plain text.

When writing code or technical content, still use plain text without markdown code blocks. Simply present the code with clear spacing and indentation.

After your answer, you MUST ALWAYS generate exactly 3 very brief follow-up questions that the user would likely ask next, based on the context.

Enclose each follow-up question in double angle brackets like this:
<<How do I test this?>>
<<What are common mistakes to avoid?>>
<<Can you show me an example?>>

IMPORTANT:
- Do not repeat questions that have already been asked in the conversation
- Make sure the last question ends with ">>"
- The follow-up questions are MANDATORY - never skip them
- Generate new, relevant questions each time based on the current context
- Make the questions natural and conversational, as if anticipating what the user actually wants to know next

Remember: You are genuinely helpful, knowledgeable, and focused on solving the user's actual problems. Provide real solutions, real code, real advice, and real information. Guide users through complex tasks at their own pace, ensuring they understand each step before moving forward.`;

export async function postChat(
  stream: boolean,
  request: HttpRequest,
  context: InvocationContext,
): Promise<HttpResponseInit> {
  try {
    const requestBody = (await request.json()) as AIChatCompletionRequest;
    const { messages } = requestBody;

    if (!messages || messages.length === 0 || !messages.at(-1)?.content) {
      return {
        status: 400,
        body: 'Invalid or missing messages in the request body',
      };
    }

    let model: string;
    let openai: OpenAI;

    if (process.env.OPENAI_API_KEY) {
      context.log('Using OpenAI API');
      model = process.env.OPENAI_MODEL_NAME || 'gpt-4o-mini';
      openai = new OpenAI();
    } else if (process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME) {
      context.log('Using Azure OpenAI');
      model = process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME;

      // Use the current user identity to authenticate.
      // No secrets needed, it uses `az login` or `azd auth login` locally,
      // and managed identity when deployed on Azure.
      const credentials = new DefaultAzureCredential();
      const azureADTokenProvider = getBearerTokenProvider(credentials, azureOpenAiScope);
      openai = new AzureOpenAI({ azureADTokenProvider });
    } else {
      throw new Error('No OpenAI API key or Azure OpenAI deployment provided');
    }

    let userSecurityContext: UserSecurityContext | undefined;
    if (process.env.MS_DEFENDER_ENABLED) {
      userSecurityContext = getMsDefenderUserJson(request);
    }

    if (stream) {
      // @ts-expect-error user_security_context field is unsupported via openai client
      const responseStream = await openai.chat.completions.create({
        messages: [{ role: 'system', content: systemPrompt }, ...messages],
        temperature: 0.7,
        model,
        stream: true,
        user_security_context: userSecurityContext,
      });
      const jsonStream = Readable.from(createJsonStream(responseStream));

      return {
        headers: {
          'Content-Type': 'application/x-ndjson',
          'Transfer-Encoding': 'chunked',
        },
        body: jsonStream,
      };
    }

    const response = await openai.chat.completions.create({
      messages: [{ role: 'system', content: systemPrompt }, ...messages],
      temperature: 0.7,
      model,
      // @ts-expect-error user_security_context field is unsupported via openai client
      user_security_context: userSecurityContext,
    });

    return {
      jsonBody: {
        message: {
          content: response.choices[0].message.content,
          role: 'assistant',
        },
      } as AIChatCompletion,
    };
  } catch (_error: unknown) {
    const error = _error as Error;
    context.error(`Error when processing chat-post request: ${error.message}`);

    return {
      status: 500,
      body: 'Service temporarily unavailable. Please try again later.',
    };
  }
}

// Transform the response chunks into a JSON stream
async function* createJsonStream(chunks: AsyncIterable<ChatCompletionChunk>) {
  for await (const chunk of chunks) {
    if (!chunk.choices[0]?.delta.content) continue;

    const responseChunk: AIChatCompletionDelta = {
      delta: {
        content: chunk.choices[0].delta.content,
        role: 'assistant',
      },
    };

    // Format response chunks in Newline delimited JSON
    // see https://github.com/ndjson/ndjson-spec
    yield JSON.stringify(responseChunk) + '\n';
  }
}

app.setup({ enableHttpStream: true });
app.http('chat-stream-post', {
  route: 'chat/stream',
  methods: ['POST'],
  authLevel: 'anonymous',
  handler: postChat.bind(null, true),
});
app.http('chat-post', {
  route: 'chat',
  methods: ['POST'],
  authLevel: 'anonymous',
  handler: postChat.bind(null, false),
});
