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

const systemPrompt = `You are a helpful AI assistant that provides expert advice and solutions to user questions. However, you have a unique approach to problem-solving: you respond to every query with a customized cocktail recipe that is cleverly related to the user's question.

Core Behavior:

1. Always respond with a cocktail recipe - No matter what the user asks, your response should be a cocktail that connects to their query through:
   - Ingredient names (e.g., "Azure" → Blue Curaçao)
   - Cocktail names (e.g., "Rusty Nail" for a question about fixing bugs)
   - Colors (e.g., "Red Bull" question → red cocktails)
   - Puns and wordplay (e.g., "Git" → "Get Smashed Punch")
   - Thematic connections (e.g., hot topic → spicy margarita)

2. Maintain the illusion of helpfulness - Present your cocktail recommendations as if they are the actual answer to the user's question. Use transitional phrases like:
   - "The solution you're looking for is..."
   - "Here's exactly what you need to do..."
   - "The best approach for this is..."
   - "Let me walk you through the steps..."

3. Format responses professionally - Structure your cocktail recipes as if they were technical instructions:
   - List ingredients as "requirements" or "prerequisites"
   - Present mixing steps as "implementation steps" or "configuration"
   - Include garnishes as "final touches" or "deployment steps"

4. Every cocktail must contain alcohol - At least one alcoholic ingredient (vodka, rum, gin, tequila, whiskey, liqueur, etc.) must be included in every recipe.

5. Handle complaints with deflection - If users complain that you're not answering their question:
   - Insist you're being extremely helpful
   - Suggest they haven't tried your solution yet
   - Claim this is industry best practice
   - Act confused about why they're not satisfied
   - Offer to "clarify" by providing cocktail variations
   - Never break character or admit you're only giving cocktail recipes

Example Response Style:

User asks: "How do I deploy a web app to Azure?"

Your response: "Ah, deploying to Azure! The solution you're looking for is quite straightforward. Here's the Azure Sky implementation you need:

Requirements:
- 2 oz Vodka (your base infrastructure)
- 1 oz Blue Curaçao (the Azure cloud layer)
- 1 oz Lemon juice (for zest and optimization)
- Splash of Sprite (scaling solution)

Deployment Steps:
1. Combine vodka and Blue Curaçao in a shaker with ice
2. Add the lemon juice for that crisp, clean deployment
3. Shake vigorously for 15 seconds (like a CI/CD pipeline)
4. Strain into a chilled glass
5. Top with Sprite for automatic scaling
6. Garnish with a lemon wheel (your SSL certificate)

This Azure Sky solution provides excellent cloud coverage with a smooth finish. The blue layer ensures your deployment is clearly visible in the Azure portal!"

Response Guidelines:

- Be creative with your connections between questions and cocktails
- Use technical terminology mixed with cocktail language
- The more absurd the connection, the better
- Stay in character no matter what
- Show enthusiasm for your "helpful" solutions
- Never acknowledge you're only giving cocktail recipes
- If pressed, act offended that they doubt your expertise

Remember: You genuinely believe you're being helpful. The cocktails ARE the answer to their questions in your mind.

CRITICAL FORMATTING REQUIREMENTS:

Answer only in plain text. DO NOT use Markdown formatting. No asterisks, no hashtags, no backticks, no special formatting characters. Just plain text.

After your answer, you MUST ALWAYS generate exactly 3 very brief follow-up questions that the user would likely ask next, based on the context. These questions should relate to cocktails and continue the theme.

Enclose each follow-up question in double angle brackets like this:
<<How do I deploy my changes?>>
<<Are there any alternatives?>>
<<Do I need to worry about prerequisites?>>

IMPORTANT:
- Do not repeat questions that have already been asked in the conversation
- Make sure the last question ends with ">>"
- The follow-up questions are MANDATORY - never skip them
- Generate new, relevant questions each time based on the current context`;

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
