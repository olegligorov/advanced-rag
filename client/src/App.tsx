import { Layout } from '@/components/layout'
import { MessageList, ChatContainer } from '@/components/chat'
import { WelcomeScreen } from '@/components/welcome'
import { useChat } from '@/hooks/useChat'
import { ThemeProvider } from '@/components/theme-provider'

const SUGGESTED_QUESTIONS = [
  "What is the kubectl command to switch my context?",
  "How do I troubleshoot a 503 Service Unavailable error when using an Nginx Ingress?",
  "What are the most common reasons a Pod remains in a Pending state?",
  "What are Pods in kubernetes?",
]

function App() {
  const { messages, input, setInput, isLoading, sendMessage } = useChat()

  const handleSubmit = () => {
    if (input.trim() && !isLoading) {
      sendMessage(input)
    }
  }

  const hasMessages = messages.length > 0

  return (
    <ThemeProvider defaultTheme="dark" storageKey="vela-theme">
      <Layout>
        {hasMessages ? (
          <MessageList messages={messages} isLoading={isLoading} />
        ) : (
          <WelcomeScreen
            questions={SUGGESTED_QUESTIONS}
            onSelectQuestion={sendMessage}
          />
        )}
        <ChatContainer
          value={input}
          onChange={setInput}
          onSubmit={handleSubmit}
          isLoading={isLoading}
        />
      </Layout>
    </ThemeProvider>
  )
}

export default App
