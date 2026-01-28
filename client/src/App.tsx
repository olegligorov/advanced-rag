import { Layout } from '@/components/layout'
import { MessageList, ChatContainer } from '@/components/chat'
import { WelcomeScreen } from '@/components/welcome'
import { useChat } from '@/hooks/useChat'
import { ThemeProvider } from '@/components/theme-provider'

const SUGGESTED_QUESTIONS = [
  "Why is my Pod stuck in CrashLoopBackOff and how do I debug it?",
  "How do I troubleshoot a 503 Service Unavailable error when using an Nginx Ingress?",
  "What are the most common reasons a Pod remains in a Pending state?",
  "Show me the kubectl command to find pods that have restarted more than 5 times in the last hour.",
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
