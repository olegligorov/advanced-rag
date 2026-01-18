import { Database } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

export function EmptyState() {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-16 text-center">
        <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
          <Database className="h-6 w-6 text-primary" />
        </div>
        <h3 className="mb-2 text-lg font-semibold">Ready to help</h3>
        <p className="text-sm text-muted-foreground max-w-md">
          Ask any question about Kubernetes and get instant answers from the official documentation with source citations
        </p>
      </CardContent>
    </Card>
  )
}
