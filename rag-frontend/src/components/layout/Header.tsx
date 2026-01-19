import { Sparkles } from 'lucide-react'

export function Header() {
  return (
    <header className="border-b bg-background/95 backdrop-blur supports-backdrop-filter:bg-background/60">
      <div className="container mx-auto max-w-5xl px-4 py-6">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center h-10 w-10 rounded-lg bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Kubernetes Documentation Assistant
            </h1>
            <p className="text-sm text-muted-foreground">
              AI-powered search with real-time streaming responses
            </p>
          </div>
        </div>
      </div>
    </header>
  )
}
