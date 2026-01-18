import { useState } from 'react'
import { queryRAGStream, type Source } from '@/services/api'

interface UseRAGQueryReturn {
  answer: string
  sources: Source[]
  isStreaming: boolean
  executeQuery: (question: string) => Promise<void>
}

/**
 * Custom hook for managing RAG query state and streaming logic.
 *
 * Encapsulates all the streaming logic, state management, and error handling
 * for RAG queries, making it easy to reuse across different components.
 *
 * Note: The number of sources (top_n) is configured in the backend and not
 * exposed to the frontend for simplicity.
 *
 * @returns Object containing query state and execution function
 *
 * @example
 * ```tsx
 * const { answer, sources, isStreaming, executeQuery } = useRAGQuery()
 *
 * const handleSubmit = (question: string) => {
 *   executeQuery(question)
 * }
 * ```
 */
export function useRAGQuery(): UseRAGQueryReturn {
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const executeQuery = async (question: string) => {
    // Reset state for new query
    setAnswer('')
    setSources([])
    setIsStreaming(true)

    try {
      // Stream the query response (top_n defaults to backend config)
      for await (const event of queryRAGStream({ question })) {
        switch (event.type) {
          case 'metadata':
            // Sources arrive first
            setSources(event.sources)
            break

          case 'chunk':
            // Answer streams incrementally
            setAnswer((prev) => prev + event.content)
            break

          case 'done':
            // Stream complete
            setIsStreaming(false)
            break

          case 'error':
            console.error('Stream error:', event.message)
            setIsStreaming(false)
            setAnswer('Error: ' + event.message)
            break
        }
      }
    } catch (error) {
      console.error('Query error:', error)
      setIsStreaming(false)
      setAnswer('Failed to process query. Please try again.')
    }
  }

  return {
    answer,
    sources,
    isStreaming,
    executeQuery,
  }
}
