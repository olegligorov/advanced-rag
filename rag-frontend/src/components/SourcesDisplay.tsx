import { cn } from '../lib/utils'
import type { Source } from '../services/api'

interface SourcesDisplayProps {
  sources: Source[]
  className?: string
}

export function SourcesDisplay({ sources, className }: SourcesDisplayProps) {
  if (sources.length === 0) {
    return null
  }

  return (
    <div className={cn('space-y-3', className)}>
      <h2 className="text-lg font-semibold text-foreground">
        Sources ({sources.length})
      </h2>

      <div className="space-y-2">
        {sources.map((source) => (
          <div
            key={source.rank}
            className={cn(
              'rounded-lg border border-border bg-card p-4',
              'hover:bg-accent/50 transition-colors'
            )}
          >
            <div className="flex items-start gap-3">
              <div
                className={cn(
                  'flex h-6 w-6 shrink-0 items-center justify-center',
                  'rounded-full bg-primary/10 text-xs font-semibold text-primary'
                )}
              >
                {source.rank}
              </div>

              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-card-foreground">
                    {source.source}
                  </span>
                </div>

                <p className="text-sm text-muted-foreground leading-relaxed">
                  {source.snippet}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
