import { AnswerDisplay } from './AnswerDisplay'
import { SourcesDisplay } from './SourcesDisplay'
import type { Source } from '@/services/api'

interface ResultsSectionProps {
  answer: string
  sources: Source[]
  isStreaming: boolean
}

/**
 * Component that displays both the answer and sources sections.
 *
 * Groups the answer and sources displays together for better organization.
 * Only renders when there's content to show.
 */
export function ResultsSection({ answer, sources, isStreaming }: ResultsSectionProps) {
  const hasContent = answer || sources.length > 0

  if (!hasContent) {
    return null
  }

  return (
    <div className="space-y-6">
      <AnswerDisplay answer={answer} isStreaming={isStreaming} />
      <SourcesDisplay sources={sources} />
    </div>
  )
}
