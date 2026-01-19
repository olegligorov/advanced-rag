# RAG Frontend

Modern React + TypeScript frontend for the Kubernetes Documentation RAG system.

## Tech Stack

- **React 19** - UI library
- **Vite** - Build tool and dev server
- **TypeScript** - Type safety
- **Tailwind CSS v4** - Styling with design tokens
- **TanStack Query** - Data fetching and caching
- **Axios** - HTTP client
- **shadcn/ui patterns** - Component patterns and utilities

## Features

- Real-time streaming responses from LLM
- Server-Sent Events (SSE) for live answer generation
- Responsive design with Tailwind CSS
- Dark mode support (via CSS variables)
- Type-safe API layer
- Source citation display

## Getting Started

### Prerequisites

- Node.js 18+ (recommended: Node 20)
- Backend server running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install
```

### Configuration

Create a `.env` file (or copy from `.env.example`):

```bash
VITE_API_URL=http://localhost:8000
```

### Development

```bash
# Start dev server (default: http://localhost:5173)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint

# Format code
npm run format
```

## Project Structure

```
src/
├── components/          # React components
│   ├── QueryForm.tsx    # Question input form
│   ├── AnswerDisplay.tsx # Streaming answer display
│   └── SourcesDisplay.tsx # Retrieved sources list
├── services/            # API layer
│   └── api.ts           # Axios client + streaming functions
├── lib/                 # Utilities
│   └── utils.ts         # cn() helper for classnames
├── App.tsx              # Main app component
├── main.tsx             # Entry point with TanStack Query provider
└── index.css            # Tailwind + theme variables
```

## API Integration

The frontend connects to two backend endpoints:

### 1. Streaming Query (Default)
```typescript
POST /api/query/stream
```
Returns Server-Sent Events with real-time answer generation.

### 2. Regular Query (Available but not used)
```typescript
POST /api/query
```
Returns complete response after generation finishes.

## Streaming Implementation

The app uses the Fetch API with async generators to handle SSE:

```typescript
for await (const event of queryRAGStream({ question, top_n: 5 })) {
  switch (event.type) {
    case 'metadata':
      // Sources arrive first (~1-2 seconds)
      setSources(event.sources)
      break
    case 'chunk':
      // Answer streams word-by-word
      setAnswer(prev => prev + event.content)
      break
    case 'done':
      // Stream complete
      setIsStreaming(false)
      break
  }
}
```

## Styling

The app uses Tailwind CSS v4 with shadcn/ui design tokens:

- CSS variables for theming (light/dark mode support)
- OKLCH color space for better color perception
- Utility-first approach
- `cn()` helper for conditional classes

## Example Usage

1. Start the backend server (see `../backend/README.md`)
2. Start the frontend: `npm run dev`
3. Open `http://localhost:5173`
4. Enter a question like "How do I set Pod resource limits?"
5. Watch as sources appear immediately, then answer streams in real-time

## TypeScript

The app is fully typed with strict TypeScript:

- API types in `services/api.ts`
- Component prop types
- Stream event types
- No `any` types

## Performance

- TanStack Query caching (5 minute stale time)
- Streaming reduces perceived latency
- Sources appear in ~1-2 seconds
- Answer streams as LLM generates
- Vite HMR for instant dev updates
