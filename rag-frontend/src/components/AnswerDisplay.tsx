import { MessageSquare, Loader2 } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'

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
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-primary" />
            <CardTitle>Answer</CardTitle>
          </div>
          {isStreaming && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Generating...</span>
            </div>
          )}
        </div>
        {!isStreaming && answer && (
          <CardDescription>AI-generated response based on documentation</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        <div
          className={cn(
            'prose prose-sm max-w-none dark:prose-invert',
            'text-card-foreground leading-relaxed',
            'whitespace-pre-wrap wrap-break-word'
          )}
        >
          {answer || (
            <span className="text-muted-foreground italic">Waiting for response...</span>
          )}
          {isStreaming && (
            <span className="inline-block w-0.5 h-5 ml-1 bg-primary animate-pulse align-middle" />
          )}
        </div>
      </CardContent>
    </Card>
  )
}
