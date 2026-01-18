# Frontend Architecture

This document describes the modular architecture of the RAG frontend application.

## Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Reusability**: Components are designed to be reusable and composable
3. **Maintainability**: Small, focused files that are easy to understand and modify
4. **Extensibility**: Easy to add new features without modifying existing code
5. **Type Safety**: Full TypeScript coverage with strict typing

## Project Structure

```
src/
├── components/              # React components
│   ├── layout/             # Layout-related components
│   │   ├── Header.tsx      # App header with branding
│   │   ├── Footer.tsx      # App footer with credits
│   │   ├── Layout.tsx      # Main layout wrapper
│   │   └── index.ts        # Barrel export
│   │
│   ├── ui/                 # shadcn/ui components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── textarea.tsx
│   │   └── ...
│   │
│   ├── QueryForm.tsx       # Question input form
│   ├── AnswerDisplay.tsx   # Streaming answer display
│   ├── SourcesDisplay.tsx  # Retrieved sources list
│   ├── ResultsSection.tsx  # Groups answer + sources
│   ├── EmptyState.tsx      # Empty state placeholder
│   └── index.ts            # Barrel export
│
├── hooks/                  # Custom React hooks
│   └── useRAGQuery.ts      # Query state management
│
├── services/               # API and external services
│   └── api.ts              # Axios client + streaming
│
├── lib/                    # Utilities
│   └── utils.ts            # Helper functions (cn, etc.)
│
├── App.tsx                 # Main app component (thin orchestrator)
├── main.tsx                # Entry point + providers
└── index.css               # Global styles + Tailwind
```

## Component Hierarchy

```
App
└── Layout
    ├── Header
    ├── main (content)
    │   ├── QueryForm
    │   ├── ResultsSection
    │   │   ├── AnswerDisplay
    │   │   └── SourcesDisplay
    │   └── EmptyState
    └── Footer
```

## Component Responsibilities

### Layout Components

#### `Layout.tsx`
- Wrapper for the entire application
- Provides consistent page structure
- Includes header, main content area, and footer

#### `Header.tsx`
- Displays app branding and title
- Shows subtitle/description
- Consistent across all pages

#### `Footer.tsx`
- Displays credits and technology info
- Provides separation from main content

### Feature Components

#### `QueryForm.tsx`
**Responsibility**: Capture user input and trigger queries

**Props**:
- `onSubmit: (question: string) => void` - Callback when form is submitted
- `isLoading: boolean` - Whether a query is in progress
- `className?: string` - Optional styling

**Features**:
- Textarea for question input
- Search button with loading state
- Form validation (non-empty question)
- Disabled state during loading

#### `AnswerDisplay.tsx`
**Responsibility**: Display LLM-generated answer with streaming support

**Props**:
- `answer: string` - The answer text (accumulates during streaming)
- `isStreaming: boolean` - Whether answer is currently streaming
- `className?: string` - Optional styling

**Features**:
- Card layout with header
- Streaming cursor animation
- Loading indicator
- Conditional description based on state

#### `SourcesDisplay.tsx`
**Responsibility**: Display retrieved source documents

**Props**:
- `sources: Source[]` - Array of source documents
- `className?: string` - Optional styling

**Features**:
- Card layout with header
- Numbered badges for ranking
- File icons and names
- Document snippets
- Separators between sources

#### `ResultsSection.tsx`
**Responsibility**: Group answer and sources together

**Props**:
- `answer: string` - Answer text
- `sources: Source[]` - Source documents
- `isStreaming: boolean` - Streaming state

**Features**:
- Conditional rendering (only shows when content exists)
- Consistent spacing between answer and sources
- Logical grouping of related content

#### `EmptyState.tsx`
**Responsibility**: Show placeholder when no results

**Features**:
- Friendly icon and message
- Clear call-to-action
- Explains what the system does

## Custom Hooks

### `useRAGQuery.ts`
**Responsibility**: Manage RAG query state and streaming logic

**Returns**:
```typescript
{
  answer: string           // Current answer text
  sources: Source[]        // Retrieved sources
  isStreaming: boolean     // Streaming status
  executeQuery: (question: string) => Promise<void>
}
```

**Features**:
- Encapsulates all streaming logic
- Manages state (answer, sources, isStreaming)
- Error handling
- SSE event processing
- Reusable across components

**Usage**:
```typescript
const { answer, sources, isStreaming, executeQuery } = useRAGQuery()

// Trigger a query
await executeQuery("What is a Pod?")
```

## Services

### `api.ts`
**Responsibility**: Handle all backend communication

**Exports**:
- `queryRAG()` - Regular non-streaming query
- `queryRAGStream()` - Streaming query with async generator
- `useStreamQuery()` - Hook-friendly streaming wrapper
- Type definitions (Source, QueryRequest, QueryResponse, etc.)

**Features**:
- Axios client with base URL configuration
- SSE parsing for streaming responses
- TypeScript types for all API interactions
- Error handling

## App Component

### `App.tsx`
**Responsibility**: Thin orchestrator that wires everything together

**Size**: ~25 lines (dramatically reduced from 120+ lines)

**Structure**:
```typescript
function App() {
  // 1. Use custom hook for query logic
  const { answer, sources, isStreaming, executeQuery } = useRAGQuery()

  // 2. Derive UI state
  const hasResults = answer || sources.length > 0
  const showEmptyState = !hasResults && !isStreaming

  // 3. Render layout with components
  return (
    <Layout>
      <QueryForm onSubmit={executeQuery} isLoading={isStreaming} />
      <ResultsSection answer={answer} sources={sources} isStreaming={isStreaming} />
      {showEmptyState && <EmptyState />}
    </Layout>
  )
}
```

**Benefits**:
- Easy to understand at a glance
- No business logic (delegated to hook)
- No layout logic (delegated to components)
- Easy to modify and extend

## Data Flow

```
User Input
    ↓
QueryForm
    ↓ (onSubmit)
App.tsx → useRAGQuery hook
    ↓ (executeQuery)
api.ts → queryRAGStream()
    ↓ (SSE events)
Backend API
    ↓ (streaming response)
useRAGQuery hook (state updates)
    ↓ (answer, sources, isStreaming)
ResultsSection
    ├→ AnswerDisplay (answer, isStreaming)
    └→ SourcesDisplay (sources)
```

## Extending the Application

### Adding a New Feature Component

1. Create component in `src/components/`
2. Export from `src/components/index.ts`
3. Use in App.tsx or other components

**Example**: Adding a query history feature

```typescript
// src/components/QueryHistory.tsx
export function QueryHistory({ queries }: { queries: string[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Recent Queries</CardTitle>
      </CardHeader>
      <CardContent>
        {queries.map((q, i) => <div key={i}>{q}</div>)}
      </CardContent>
    </Card>
  )
}

// src/components/index.ts
export { QueryHistory } from './QueryHistory'

// App.tsx
import { QueryHistory } from '@/components'

function App() {
  const [queryHistory, setQueryHistory] = useState<string[]>([])
  // ... rest of component

  return (
    <Layout>
      <QueryHistory queries={queryHistory} />
      {/* ... other components */}
    </Layout>
  )
}
```

### Adding a New Hook

1. Create hook in `src/hooks/`
2. Follow naming convention: `use[Feature].ts`
3. Document with JSDoc comments

**Example**: Adding a favorites feature

```typescript
// src/hooks/useFavorites.ts
export function useFavorites() {
  const [favorites, setFavorites] = useState<string[]>([])

  const addFavorite = (query: string) => {
    setFavorites(prev => [...prev, query])
  }

  const removeFavorite = (query: string) => {
    setFavorites(prev => prev.filter(q => q !== query))
  }

  return { favorites, addFavorite, removeFavorite }
}
```

### Adding a New API Endpoint

1. Add types to `src/services/api.ts`
2. Create API function
3. Use in components via hooks

**Example**: Adding a feedback endpoint

```typescript
// src/services/api.ts
export interface FeedbackRequest {
  query: string
  helpful: boolean
}

export const submitFeedback = async (feedback: FeedbackRequest) => {
  const response = await apiClient.post('/api/feedback', feedback)
  return response.data
}
```

## Best Practices

### Component Design
- Keep components small (<100 lines)
- Single responsibility per component
- Accept props for configuration
- Emit events via callbacks
- Use TypeScript for all props

### State Management
- Use custom hooks for complex state logic
- Keep state close to where it's used
- Lift state only when necessary
- Use context for deeply nested props

### Styling
- Use shadcn/ui components where possible
- Consistent use of design tokens
- Tailwind utility classes
- `cn()` helper for conditional classes

### File Organization
- Group related files in folders
- Use index.ts for barrel exports
- Keep flat structure (max 2 levels deep)
- Name files after their default export

### Type Safety
- Define interfaces for all props
- Type all API responses
- Use TypeScript strict mode
- Avoid `any` types

## Testing Strategy (Future)

### Unit Tests
- Test hooks in isolation
- Test utility functions
- Mock API calls

### Component Tests
- Test rendering
- Test user interactions
- Test prop variations

### Integration Tests
- Test full user flows
- Test API integration
- Test error scenarios

## Performance Considerations

### Current Optimizations
- TanStack Query caching (5 min)
- Streaming reduces perceived latency
- Code splitting via Vite
- HMR for development

### Future Optimizations
- React.memo for expensive components
- useMemo for derived values
- Lazy loading for routes
- Virtual scrolling for long lists

## Accessibility

### Current Support
- Semantic HTML elements
- ARIA labels on interactive elements
- Keyboard navigation (via shadcn)
- Focus management

### Future Improvements
- Screen reader announcements
- Skip navigation links
- High contrast mode
- Reduced motion support
