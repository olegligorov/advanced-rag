import { Separator } from '@/components/ui/separator'

export function Footer() {
  return (
    <footer className="mt-16 border-t">
      <div className="container mx-auto max-w-5xl px-4 py-6">
        <Separator className="mb-6" />
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          Some Footer
        </div>
      </div>
    </footer>
  )
}
