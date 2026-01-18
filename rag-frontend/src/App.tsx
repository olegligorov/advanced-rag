import { Layout } from '@/components/layout/Layout'
import { QueryForm } from '@/components/QueryForm'
import { ResultsSection } from '@/components/ResultsSection'
import { EmptyState } from '@/components/EmptyState'
import { useRAGQuery } from '@/hooks/useRAGQuery'
import { ThemeProvider } from './components/theme-provider'

function App() {
  const { answer, sources, isStreaming, executeQuery } = useRAGQuery()

  const hasResults = answer || sources.length > 0
  const showEmptyState = !hasResults && !isStreaming

  return (
    <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
      <Layout>
        <div className="space-y-8">
          <QueryForm onSubmit={executeQuery} isLoading={isStreaming} />

          <ResultsSection
            answer={answer}
            sources={sources}
            isStreaming={isStreaming}
          />

          {showEmptyState && <EmptyState />}
        </div>
      </Layout>
    </ThemeProvider>
  )
}

export default App
