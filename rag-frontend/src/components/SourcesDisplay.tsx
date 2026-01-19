import { FileText, BookOpen } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import type { Source } from '@/services/api'

interface SourcesDisplayProps {
  sources: Source[]
  className?: string
}

export function SourcesDisplay({ sources, className }: SourcesDisplayProps) {
  if (sources.length === 0) {
    return null
  }

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-primary" />
          <CardTitle>Sources</CardTitle>
        </div>
        <CardDescription>
          Retrieved {sources.length} relevant {sources.length === 1 ? 'document' : 'documents'} from documentation
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {sources.map((source, index) => (
          <div key={source.rank}>
            {index > 0 && <Separator className="mb-4" />}
            <div className="flex items-start gap-4">
              <Badge variant="outline" className="shrink-0 h-6 w-6 flex items-center justify-center p-0 rounded-full">
                {source.rank}
              </Badge>

              <div className="flex-1 space-y-2 min-w-0">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="text-sm font-medium text-card-foreground truncate">
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
      </CardContent>
    </Card>
  )
}
