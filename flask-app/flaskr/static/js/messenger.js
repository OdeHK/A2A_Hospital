const sendBtn = document.getElementById('sendBtn');
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chat-messages');
sendBtn.addEventListener('click', sendMessage);
messageInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendMessage();
});

console.log(settings)

function sendMessage() {
  const text = messageInput.value.trim();
  if (text === '') return;

  // Create message bubble
  const message = document.createElement('div');
  message.classList.add('message', 'sent');
  message.textContent = text;
  chatMessages.appendChild(message);

  // Auto-scroll
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Clear input
  messageInput.value = '';

  // Simulate a reply
  setTimeout(() => {
    const reply = document.createElement('div');
    reply.classList.add('message', 'received');
    reply.textContent = "Chat: " + text;
    chatMessages.appendChild(reply);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }, 800);
}
