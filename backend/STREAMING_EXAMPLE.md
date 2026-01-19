# Streaming API Usage

## Backend Endpoints

### 1. Regular Query (Non-streaming)
```
POST /api/query
```
Returns complete response after generation finishes.

### 2. Streaming Query (NEW)
```
POST /api/query/stream
```
Returns Server-Sent Events (SSE) stream with real-time answer generation.

## Stream Response Format

The streaming endpoint sends events in this order:

1. **Metadata Event** (first):
```json
{
  "type": "metadata",
  "sources": [
    {
      "rank": 1,
      "source": "pods.md",
      "snippet": "A Pod is the smallest..."
    }
  ],
  "question": "What is a Pod?"
}
```

2. **Chunk Events** (multiple):
```json
{
  "type": "chunk",
  "content": "A Pod is"
}
```

3. **Done Event** (final):
```json
{
  "type": "done"
}
```

## Frontend Integration Examples

### Using Fetch API (Vanilla JavaScript)

```javascript
async function streamQuery(question, topN = 5) {
  const response = await fetch('http://localhost:8000/api/query/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, top_n: topN })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let answer = '';
  let sources = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    // Decode the chunk
    const chunk = decoder.decode(value, { stream: true });

    // SSE format: "data: {...}\n\n"
    const lines = chunk.split('\n\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));

        switch (data.type) {
          case 'metadata':
            sources = data.sources;
            console.log('Sources:', sources);
            break;

          case 'chunk':
            answer += data.content;
            console.log('Chunk:', data.content);
            // Update UI here: display answer incrementally
            break;

          case 'done':
            console.log('Stream complete!');
            console.log('Final answer:', answer);
            return { answer, sources };

          case 'error':
            console.error('Error:', data.message);
            throw new Error(data.message);
        }
      }
    }
  }
}

// Usage
streamQuery('How do I set Pod resource limits?')
  .then(result => console.log('Done:', result))
  .catch(err => console.error('Error:', err));
```

### Using React with useEffect

```typescript
import { useState, useEffect } from 'react';

interface Source {
  rank: number;
  source: string;
  snippet: string;
}

function StreamingQueryComponent() {
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<Source[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const queryWithStreaming = async (question: string) => {
    setIsStreaming(true);
    setAnswer('');
    setSources([]);

    try {
      const response = await fetch('http://localhost:8000/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_n: 5 })
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            switch (data.type) {
              case 'metadata':
                setSources(data.sources);
                break;

              case 'chunk':
                setAnswer(prev => prev + data.content);
                break;

              case 'done':
                setIsStreaming(false);
                break;

              case 'error':
                console.error('Error:', data.message);
                setIsStreaming(false);
                break;
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      setIsStreaming(false);
    }
  };

  return (
    <div>
      <button
        onClick={() => queryWithStreaming('How do I set Pod limits?')}
        disabled={isStreaming}
      >
        {isStreaming ? 'Streaming...' : 'Ask Question'}
      </button>

      <div>
        <h3>Answer:</h3>
        <p>{answer || 'Waiting for answer...'}</p>
      </div>

      <div>
        <h3>Sources:</h3>
        <ul>
          {sources.map(source => (
            <li key={source.rank}>
              {source.rank}. {source.source}: {source.snippet}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

### Using EventSource API (Alternative)

```javascript
function streamQueryWithEventSource(question, topN = 5) {
  return new Promise((resolve, reject) => {
    const url = new URL('http://localhost:8000/api/query/stream');

    // EventSource doesn't support POST, so you'd need to adjust the backend
    // or use fetch API approach above
    const eventSource = new EventSource(url);

    let answer = '';
    let sources = [];

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case 'metadata':
          sources = data.sources;
          break;

        case 'chunk':
          answer += data.content;
          // Update UI
          break;

        case 'done':
          eventSource.close();
          resolve({ answer, sources });
          break;

        case 'error':
          eventSource.close();
          reject(new Error(data.message));
          break;
      }
    };

    eventSource.onerror = (error) => {
      eventSource.close();
      reject(error);
    };
  });
}
```

## Testing with curl

```bash
# Test streaming endpoint
curl -N -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Kubernetes Pod?", "top_n": 5}'
```

The `-N` flag disables buffering to see the stream in real-time.

## Benefits of Streaming

1. **Better UX**: Users see responses immediately instead of waiting 5-10 seconds
2. **Perceived Performance**: Streaming feels much faster even if total time is the same
3. **Progressive Enhancement**: Can show partial answers even if connection drops
4. **Lower Time to First Byte (TTFB)**: Sources appear immediately after retrieval

## Comparison

| Feature | Regular Endpoint | Streaming Endpoint |
|---------|-----------------|-------------------|
| Time to first content | 5-10 seconds | ~1 second (sources) |
| User experience | "Frozen" UI | Live updates |
| Connection type | Request-response | Server-Sent Events |
| Total time | Same | Same |
| Use case | Batch/background | Interactive UI |
