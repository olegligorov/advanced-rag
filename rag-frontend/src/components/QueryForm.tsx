import { useState } from 'react'
import { cn } from '../lib/utils'

interface QueryFormProps {
  onSubmit: (question: string, topN: number) => void
  isLoading: boolean
  className?: string
}

export function QueryForm({ onSubmit, isLoading, className }: QueryFormProps) {
  const [question, setQuestion] = useState('')
  const [topN, setTopN] = useState(5)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (question.trim()) {
      onSubmit(question, topN)
    }
  }

  return (
    <form onSubmit={handleSubmit} className={cn('space-y-4', className)}>
      <div className="space-y-2">
        <label htmlFor="question" className="text-sm font-medium text-foreground">
          Ask a question
        </label>
        <textarea
          id="question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g., How do I set Pod resource limits?"
          disabled={isLoading}
          rows={3}
          className={cn(
            'w-full rounded-lg border border-input bg-background px-3 py-2',
            'text-sm text-foreground placeholder:text-muted-foreground',
            'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'resize-none'
          )}
        />
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <label htmlFor="top_n" className="text-sm font-medium text-foreground">
            Top results:
          </label>
          <input
            id="top_n"
            type="number"
            min="1"
            max="20"
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            disabled={isLoading}
            className={cn(
              'w-16 rounded-md border border-input bg-background px-2 py-1',
              'text-sm text-foreground',
              'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading || !question.trim()}
          className={cn(
            'ml-auto rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground',
            'hover:bg-primary/90 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'transition-colors'
          )}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </div>
    </form>
  )
}
