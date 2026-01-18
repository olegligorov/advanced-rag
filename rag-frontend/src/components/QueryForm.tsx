import { useState, type KeyboardEvent } from 'react'
import { ArrowUp, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface QueryFormProps {
  onSubmit: (question: string) => void
  isLoading: boolean
  className?: string
}

export function QueryForm({ onSubmit, isLoading, className }: QueryFormProps) {
  const [question, setQuestion] = useState('')

  const handleSubmit = () => {
    if (question.trim() && !isLoading) {
      onSubmit(question)
      setQuestion('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
    // Allow Shift+Enter for new line (default textarea behavior)
  }

  return (
    <Card className={cn('p-6', className)}>
      <div className="relative">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about Kubernetes... (Shift+Enter for new line)"
          disabled={isLoading}
          rows={3}
          className={cn(
            'w-full rounded-lg border border-input bg-background pl-4 pr-14 py-3',
            'text-sm text-foreground placeholder:text-muted-foreground',
            'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'resize-none'
          )}
        />

        <Button
          type="button"
          onClick={handleSubmit}
          disabled={isLoading || !question.trim()}
          size="icon"
          className={cn(
            'absolute bottom-4 right-3 h-9 w-9 rounded-xl',
            'transition-all duration-200',
            !question.trim() && 'opacity-50'
          )}
        >
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <ArrowUp className="h-4 w-4" />
          )}
        </Button>
      </div>

      <p className="mt-2 text-xs text-muted-foreground">
        Press <kbd className="px-1.5 py-0.5 text-xs font-semibold bg-muted rounded">Enter</kbd> to send,
        <kbd className="ml-1 px-1.5 py-0.5 text-xs font-semibold bg-muted rounded">Shift + Enter</kbd> for new line
      </p>
    </Card>
  )
}
