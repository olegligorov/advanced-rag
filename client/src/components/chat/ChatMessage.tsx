import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message'
import {
  Sources,
  SourcesTrigger,
  SourcesContent,
  Source,
} from '@/components/ai-elements/sources'
import type { UIMessage } from 'ai'

interface Source {
  title: string
  snippet: string
}

interface ChatMessageProps {
  role: UIMessage['role']
  content: string
  sources?: Source[]
  isStreaming?: boolean
}

export function ChatMessage({ role, content, sources, isStreaming }: ChatMessageProps) {
  return (
    <Message from={role} className="mb-4">
      <MessageContent>
        {sources && sources.length > 0 && (
          <Sources>
            <SourcesTrigger count={sources.length} />
            <SourcesContent>
              {sources.map((source, index) => {
                // Create URL to backend endpoint that serves the file
                const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
                const fileUrl = `${apiUrl}/api/file?path=${encodeURIComponent(source.title)}`

                return (
                  <Source key={index} href={fileUrl} title={source.title}>
                    <div className="flex flex-col gap-1">
                      <span className="font-medium text-xs">{source.title}</span>
                      <span className="text-muted-foreground text-xs line-clamp-2">
                        {source.snippet}
                      </span>
                    </div>
                  </Source>
                )
              })}
            </SourcesContent>
          </Sources>
        )}
        <MessageResponse>
          {content}
        </MessageResponse>
        {isStreaming && <span className="inline-block w-0.5 h-4 ml-1 bg-primary animate-pulse align-middle" />}
      </MessageContent>
    </Message>
  )
}
