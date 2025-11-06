const sendBtn = document.getElementById('sendBtn');
const messageInput = document.getElementById('messageInput');
const chatMessages = document.getElementById('chat-messages');
const audioMessages =  document.getElementById('sendAudio');
sendBtn.addEventListener('click', sendMessage);
audioMessages.addEventListener('click' ,  sendAudio)
messageInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendMessage();
});

console.log(settings)
function sendAudio() {
  const text = messageInput.value.trim();
  const message = document.createElement('div');
  message.classList.add('message', 'sent');
  message.textContent = text;
  chatMessages.appendChild(message);

  // Auto-scroll
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Clear input
  messageInput.value = '';
  fetch('/messenger/voice-chat' , {
    method : 'POST' , 
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: text }),
  }).then(response => {
    return response.json();
  }).then(data => {
    
    const reply = document.createElement('div');
    reply.classList.add('message', 'received' , 'audio-response');
    const audioElement = document.createElement("audio");
    audioElement.controls = true;
    const sourceElement = document.createElement("source");
    sourceElement.src = data['result'];
    sourceElement.type = "audio/wav";

    audioElement.appendChild(sourceElement);
    reply.appendChild(audioElement);
    chatMessages.appendChild(reply);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  })
}
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
