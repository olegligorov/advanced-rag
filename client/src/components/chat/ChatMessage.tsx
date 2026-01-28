import { cn } from '@/lib/utils'
import { Bot, User } from 'lucide-react'

interface Source {
  title: string
  snippet: string
}

interface ChatMessageProps {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  isStreaming?: boolean
}

export function ChatMessage({ role, content, sources, isStreaming }: ChatMessageProps) {
  return (
    <div
      className={cn(
        'flex gap-4 p-4 rounded-xl',
        role === 'user' ? 'bg-secondary/50' : 'bg-card border border-border'
      )}
    >
      <div
        className={cn(
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg',
          role === 'user'
            ? 'bg-muted text-muted-foreground'
            : 'bg-primary text-primary-foreground'
        )}
      >
        {role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div className="flex-1 space-y-4">
        {sources && sources.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Sources
            </p>
            <div className="grid gap-2">
              {sources.map((source, index) => (
                <div key={index} className="p-3 rounded-lg bg-secondary/30 border border-border/50">
                  <p className="text-xs font-medium text-foreground mb-1">{source.title}</p>
                  <p className="text-xs text-muted-foreground line-clamp-2">
                    {source.snippet}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          {role === 'user' ? 'Question' : 'Answer'}
        </p>
        <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">
          {content}
          {isStreaming && <span className="inline-block w-0.5 h-4 ml-1 bg-primary animate-pulse align-middle" />}
        </p>
      </div>
    </div>
  )
}
