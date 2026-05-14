import { beforeEach, expect, mock, test } from "bun:test"
import { Hono } from "hono"

import type { AnthropicMessagesPayload } from "../src/routes/messages/anthropic-types"
import type {
  CreateResponsesReturn,
  ResponseCompletedEvent,
  ResponseCreatedEvent,
  ResponseTextDeltaEvent,
  ResponsesPayload,
  ResponsesResult,
} from "../src/services/copilot/create-responses"

const actualCreateResponsesModule = await import(
  "../src/services/copilot/create-responses"
)
const actualTokenUsageModule = await import("../src/lib/token-usage")

type MockStreamChunk = {
  event?: string
  data?: string
}

let streamedChunks: Array<MockStreamChunk> = []

const toAsyncIterable = (
  chunks: Array<MockStreamChunk>,
): CreateResponsesReturn => {
  return {
    async *[Symbol.asyncIterator]() {
      for (const chunk of chunks) {
        yield await Promise.resolve(chunk)
      }
    },
  } as unknown as CreateResponsesReturn
}

const createResponses = mock(
  (_payload: ResponsesPayload): Promise<CreateResponsesReturn> => {
    return Promise.resolve(toAsyncIterable(streamedChunks))
  },
)

const createCopilotTokenUsageRecorder = mock(() => {
  return () => {}
})

await mock.module("~/services/copilot/create-responses", () => ({
  ...actualCreateResponsesModule,
  createResponses,
}))

await mock.module("~/lib/token-usage", () => ({
  ...actualTokenUsageModule,
  createCopilotTokenUsageRecorder,
}))

const { handleWithResponsesApi } = await import(
  "../src/routes/messages/api-flows"
)

const logger = {
  debug: () => {},
  warn: () => {},
  error: () => {},
} as unknown as Parameters<typeof handleWithResponsesApi>[2]["logger"]

const createPayload = (): AnthropicMessagesPayload => ({
  model: "gpt-test",
  max_tokens: 128,
  stream: true,
  messages: [{ role: "user", content: "hello" }],
})

const createResponsesResult = (
  overrides: Partial<ResponsesResult> = {},
): ResponsesResult => ({
  id: "resp_123",
  object: "response",
  created_at: 0,
  model: "gpt-test",
  output: [],
  output_text: "",
  status: "completed",
  usage: {
    input_tokens: 12,
    output_tokens: 4,
    total_tokens: 16,
    input_tokens_details: {
      cached_tokens: 0,
    },
  },
  error: null,
  incomplete_details: null,
  instructions: null,
  metadata: null,
  parallel_tool_calls: false,
  temperature: null,
  tool_choice: null,
  tools: [],
  top_p: null,
  ...overrides,
})

const createCreatedEvent = (): ResponseCreatedEvent => ({
  type: "response.created",
  sequence_number: 1,
  response: createResponsesResult({ status: "in_progress" }),
})

const createCompletedEvent = (): ResponseCompletedEvent => ({
  type: "response.completed",
  sequence_number: 3,
  response: createResponsesResult({
    output: [
      {
        id: "msg_1",
        type: "message",
        role: "assistant",
        status: "completed",
        content: [
          {
            type: "output_text",
            text: "hello",
            annotations: [],
          },
        ],
      },
    ],
    output_text: "hello",
  }),
})

const createTextDeltaEvent = (): ResponseTextDeltaEvent => ({
  type: "response.output_text.delta",
  sequence_number: 2,
  output_index: 0,
  content_index: 0,
  item_id: "msg_1",
  delta: "hello",
})

const createApp = () => {
  const app = new Hono()
  app.get("/", (c) => {
    return handleWithResponsesApi(c, createPayload(), {
      logger,
      requestId: "request-1",
    })
  })
  return app
}

beforeEach(() => {
  streamedChunks = []
  createResponses.mockClear()
  createCopilotTokenUsageRecorder.mockClear()
})

test("messages Responses flow tolerates split JSON stream events", async () => {
  const textDelta = JSON.stringify(createTextDeltaEvent())
  const splitIndex = textDelta.indexOf("hello") + 3

  streamedChunks = [
    {
      event: "response.created",
      data: JSON.stringify(createCreatedEvent()),
    },
    {
      event: "response.output_text.delta",
      data: textDelta.slice(0, splitIndex),
    },
    {
      event: "response.output_text.delta",
      data: textDelta.slice(splitIndex),
    },
    {
      event: "response.completed",
      data: JSON.stringify(createCompletedEvent()),
    },
  ]

  const response = await createApp().request("/")

  expect(response.status).toBe(200)
  expect(createResponses).toHaveBeenCalledTimes(1)

  const body = await response.text()

  expect(body).toContain("event: message_start")
  expect(body).toContain('"text":"hello"')
  expect(body).toContain("event: message_stop")
})

test("messages Responses flow skips malformed JSON events and continues streaming", async () => {
  streamedChunks = [
    {
      event: "response.created",
      data: JSON.stringify(createCreatedEvent()),
    },
    {
      event: "response.output_text.delta",
      data: '{"type":"response.output_text.delta","delta":"broken',
    },
    {
      event: "response.completed",
      data: JSON.stringify(createCompletedEvent()),
    },
  ]

  const response = await createApp().request("/")

  expect(response.status).toBe(200)

  const body = await response.text()

  expect(body).toContain("event: message_start")
  expect(body).toContain("event: message_stop")
  expect(body).not.toContain("Responses stream ended without completion")
})
