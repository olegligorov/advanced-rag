import { cn } from '../lib/utils'

interface AnswerDisplayProps {
  answer: string
  isStreaming: boolean
  className?: string
}

export function AnswerDisplay({ answer, isStreaming, className }: AnswerDisplayProps) {
  if (!answer && !isStreaming) {
    return null
  }

  return (
    <div className={cn('space-y-3', className)}>
      <div className="flex items-center gap-2">
        <h2 className="text-lg font-semibold text-foreground">Answer</h2>
        {isStreaming && (
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <div className="h-2 w-2 animate-pulse rounded-full bg-primary" />
            <span>Generating...</span>
          </div>
        )}
      </div>

      <div
        className={cn(
          'rounded-lg border border-border bg-card p-4',
          'text-sm text-card-foreground leading-relaxed',
          'whitespace-pre-wrap wrap-break-word'
        )}
      >
        {answer || (
          <span className="text-muted-foreground italic">Waiting for response...</span>
        )}
        {isStreaming && <span className="inline-block w-1 h-4 ml-1 bg-primary animate-pulse" />}
      </div>
    </div>
  )
}
