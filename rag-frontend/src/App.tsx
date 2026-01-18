import { useState } from 'react'
import { QueryForm } from './components/QueryForm'
import { AnswerDisplay } from './components/AnswerDisplay'
import { SourcesDisplay } from './components/SourcesDisplay'
import { queryRAGStream, type Source } from './services/api'

function App() {
  // const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [isStreaming, setIsStreaming] = useState(false)

  const handleQuery = async (userQuestion: string, topN: number) => {
    // setQuestion(userQuestion)
    setAnswer('')
    setSources([])
    setIsStreaming(true)

    try {
      for await (const event of queryRAGStream({
        question: userQuestion,
        top_n: topN,
      })) {
        switch (event.type) {
          case 'metadata':
            setSources(event.sources)
            break

          case 'chunk':
            setAnswer((prev) => prev + event.content)
            break

          case 'done':
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

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-4xl px-4 py-8 space-y-8">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold text-foreground">
            Kubernetes Documentation RAG
          </h1>
          <p className="text-muted-foreground">
            Ask questions about Kubernetes and get answers from official documentation
          </p>
        </div>

        {/* Query Form */}
        <QueryForm onSubmit={handleQuery} isLoading={isStreaming} />

        {/* Results */}
        {(answer || sources.length > 0) && (
          <div className="space-y-6">
            {/* Answer Section */}
            <AnswerDisplay answer={answer} isStreaming={isStreaming} />

            {/* Sources Section */}
            <SourcesDisplay sources={sources} />
          </div>
        )}

        {/* Empty State */}
        {!answer && !isStreaming && sources.length === 0 && (
          <div className="rounded-lg border border-dashed border-border p-12 text-center">
            <p className="text-muted-foreground">
              Enter a question above to get started
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
