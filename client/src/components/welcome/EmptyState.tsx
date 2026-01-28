import { Sparkles } from 'lucide-react'

export function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center text-center py-16 px-4">
      <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 mb-6">
        <Sparkles className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-2xl font-semibold text-foreground mb-2 text-balance">
        Ask anything about your knowledge base
      </h2>
      <p className="text-muted-foreground max-w-md text-balance leading-relaxed">
        I can search through your documents and provide accurate answers with source citations. Try
        one of the suggested questions below.
      </p>
    </div>
  )
}
