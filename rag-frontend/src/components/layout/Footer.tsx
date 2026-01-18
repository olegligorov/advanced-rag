import { Separator } from '@/components/ui/separator'

export function Footer() {
  return (
    <footer className="mt-16 border-t">
      <div className="container mx-auto max-w-5xl px-4 py-6">
        <Separator className="mb-6" />
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <p>Powered by RAG (Retrieval-Augmented Generation)</p>
          <p>Built with React + Vite + shadcn/ui</p>
        </div>
      </div>
    </footer>
  )
}
