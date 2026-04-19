# Hermes Empty Response Analysis

## Summary

Hermes is still reporting `Empty response from model` even though the proxy logs show:

- `x-stainless-async: false`
- `stream=False`
- `200 OK` responses
- a non-empty response body such as `hello :) how can i help?`

That means the issue is likely **not** SSE transport anymore. The proxy is returning a normal JSON response, but Hermes is probably rejecting it during response parsing or agent-loop evaluation.

## What the logs show

Observed request pattern:

- Hermes sends `user-agent: OpenAI/Python 2.32.0`
- Hermes sends `x-stainless-async: false`
- Proxy detects `agent=hermes`
- Proxy logs `stream=False`
- Proxy returns JSON successfully
- Proxy logs real text content in the response
- Hermes still retries 3 times and then reports an empty response

This strongly suggests the failure happens **after** the proxy sends the payload.

## Likely root causes

### 1. Response shape mismatch

Hermes may not accept the exact OpenAI chat-completions shape the proxy returns.

Possible mismatch points:

- `choices[0].message.content`
- `choices[0].finish_reason`
- `choices[0].message.role`
- presence or absence of `tool_calls`
- model name / provider expectations

Even if the JSON is valid OpenAI-compatible output, Hermes may have stricter expectations.

### 2. Tool-call expectations

Hermes may be operating in a tool-driven loop and may ignore plain `content` if it expects:

- `tool_calls`
- a specific tool name
- a specific tool result structure

If Hermes sees a response without the tool pattern it expects, it may treat the turn as empty.

### 3. Client-side response filtering

The Hermes client may parse the response successfully but then discard it because:

- it does not match the expected model backend contract
- it does not contain a recognized completion state
- it is considered incomplete by the agent runtime

So the proxy may be returning content, but the client layer may not be using it.

## Why streaming is probably not the cause now

Earlier, streaming was a reasonable suspect. But the current logs show:

- `stream=False` in proxy logs
- `Content-Type: application/json`
- a full response body is generated

So the problem is probably **not SSE vs JSON transport** anymore.

## Best interpretation

The proxy is likely returning a response that is syntactically valid but **semantically incompatible** with Hermes’ expectations.

In other words:

- the proxy says: “here is a completed assistant response”
- Hermes says: “I can’t use this as a valid agent turn”

## Recommended next debugging steps

### 1. Capture the exact raw response body

Log the full JSON body returned to Hermes for one request.

Check whether Hermes receives:

- plain assistant content only
- a tool-call response
- a hybrid content + tool_calls response

### 2. Reduce the Hermes response to the minimum

Try the simplest possible payload for Hermes:

- `object: "chat.completion"`
- `choices[0].message.role: "assistant"`
- `choices[0].message.content: "hello"`
- `choices[0].finish_reason: "stop"`
- no `tool_calls`

If that still retries, the issue is probably in Hermes’ parser or adapter layer.

### 3. Verify whether Hermes expects tool calls

If Hermes is meant to run in a tool-loop mode, then the proxy may need to simulate a tool call that Hermes actually recognizes.

If Hermes is not consuming tool calls, then tool-based responses may be the reason it looks empty.

### 4. Separate response modes by agent

A robust long-term design is:

- Roo: tool-call simulation
- Hermes: plain assistant content only

That keeps each agent on the response shape it actually understands.

## Practical fix options

### Option A: Plain text only for Hermes

Return a minimal assistant response with:

- `content`
- `finish_reason="stop"`
- no `tool_calls`

This is the simplest option.

### Option B: Tool-call only for Hermes

Return a simulated tool call with:

- `finish_reason="tool_calls"`
- `content=None`
- `tool_calls=[...]`

Only use this if Hermes actually consumes tool calls.

### Option C: Agent-specific response contract

Keep Hermes and Roo separate:

- Hermes gets clean OpenAI chat-completion text
- Roo gets tool-call-oriented responses

This is likely the cleanest architecture if both agents must be supported.

## Conclusion

The current evidence points away from streaming and toward **response-contract incompatibility**.

Hermes is receiving data, but not accepting it as a valid assistant turn. The next step is to inspect the exact JSON payload Hermes receives and reduce it to the simplest possible response shape.
