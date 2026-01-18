import { Layout } from '@/components/layout'
import { MessageList, ChatContainer } from '@/components/chat'
import { WelcomeScreen } from '@/components/welcome'
import { useChat } from '@/hooks/useChat'
import { ThemeProvider } from '@/components/theme-provider'

const SUGGESTED_QUESTIONS = [
  'What are the main features of our product?',
  'How do I set up authentication?',
  'Explain the pricing structure',
  'What integrations are available?',
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
    <ThemeProvider defaultTheme="dark" storageKey="askbase-theme">
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
