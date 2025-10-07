class Chatbox {
  constructor() {
    this.args = {
      openButton: document.querySelector('.chatbox__button button'),
      chatBox: document.querySelector('.chatbox__support'),
      sendButton: document.querySelector('.send__button'),
      input: document.querySelector('.chatbox__input'),
      quickReplies: document.querySelectorAll('.quick-btn')
    };
    this.state = false;
    this.messages = [];
    this._init();

    // Debug: báo trạng thái marked ngay sau khi khởi tạo (an toàn)
    console.log('[Chatbox] initialized. marked available?', typeof marked !== 'undefined');
  }

  _init() {
    const { openButton, chatBox, sendButton, input, quickReplies } = this.args;

    openButton.addEventListener('click', () => this.toggleState(chatBox));
    sendButton.addEventListener('click', () => this.onSendButton());
    input.addEventListener('keyup', (e) => { if (e.key === 'Enter') this.onSendButton(); });
    quickReplies.forEach(btn => btn.addEventListener('click', () => {
      input.value = btn.dataset.value;
      this.onSendButton();
    }));
  }

  toggleState(chatbox) {
    this.state = !this.state;
    chatbox.classList.toggle('chat-open');
    if (this.state && !this._welcomed) {
      this._welcomed = true;
      this._pushBotMessage("Xin chào! Mình là trợ lý y tế, hôm nay mình có thể giúp gì cho bạn?");
    }
  }

  _pushUserMessage(text) {
    this.messages.push({ name: 'User', message: text });
    this.updateChatText();
  }
  _pushBotMessage(text) {
    this.messages.push({ name: 'Bot', message: text });
    this.updateChatText();
  }
  _pushBotTyping() {
    this.messages.push({ name: 'Bot', typing: true });
    this.updateChatText();
  }
  _replaceBotTypingWithMessage(text) {
    const idx = this.messages.findIndex(m => m.typing);
    if (idx !== -1) this.messages.splice(idx, 1);
    this.messages.push({ name: 'Bot', message: text });
    this.updateChatText();
  }

  onSendButton() {
    const input = this.args.input;
    const text = input.value.trim();
    if (!text) return;
    this._pushUserMessage(text);
    input.value = '';
    this._pushBotTyping();

    fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    })
      .then(async r => {
        // Debug: log status and raw response text for inspection
        const contentType = r.headers.get('content-type') || '';
        let body;
        try {
          if (contentType.includes('application/json')) {
            body = await r.json();
          } else {
            body = await r.text();
          }
        } catch (e) {
          console.error('[Chatbox] failed to parse response body', e);
          body = await r.text().catch(() => null);
        }
        console.log('[Chatbox] /chat response (parsed):', body);
        return body;
      })
      .then(r => {
        // r might be an object or text; try to extract answer
        let answer = "Xin lỗi, hiện không có câu trả lời.";
        if (!r) {
          answer = "⚠️ Lỗi: phản hồi rỗng từ server.";
        } else if (typeof r === 'object') {
          // nếu server trả JSON
          answer = r.answer || r.final_text || JSON.stringify(r);
        } else if (typeof r === 'string') {
          answer = r;
        }
        console.log('[Chatbox] using answer:', answer);
        this._replaceBotTypingWithMessage(answer);
      })
      .catch(err => {
        console.error('[Chatbox] fetch error:', err);
        this._replaceBotTypingWithMessage("⚠️ Lỗi kết nối server.");
      });
  }

  // Cải tiến: render Markdown nếu có, in ra debug
  updateChatText() {
    const chatbox = this.args.chatBox;
    const container = chatbox.querySelector('.chatbox__messages');
    const botAvatar = chatbox.dataset.botAvatar;
    const userAvatar = chatbox.dataset.userAvatar;

    container.innerHTML = this.messages.map(m => {
      const isBot = m.name === 'Bot';
      const rowClass = 'message-row ' + (isBot ? 'bot' : 'user');
      const bubbleClass = 'messages__item ' + (isBot ? 'messages__item--bot' : 'messages__item--user');
      const avatar = `<div class="avatar"><img src="${isBot ? botAvatar : userAvatar}" alt="${m.name}"></div>`;

      let contentHtml;
      if (m.typing) {
        contentHtml = `<span class="typing"><span></span><span></span><span></span></span>`;
      } else {
        // đảm bảo m.message là string
        let raw = m.message;
        if (typeof raw !== 'string') {
          try { raw = JSON.stringify(raw); } catch(e) { raw = String(raw); }
        }
        // debug: log raw message occasionally
        console.log('[Chatbox] rendering message raw:', raw.slice ? raw.slice(0, 500) : raw);

        try {
          if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
            // use marked to parse markdown -> HTML
            contentHtml = marked.parse(raw || '');
          } else {
            // fallback nhẹ: convert newlines + bold + bullets
            contentHtml = (raw || "")
              .replace(/\n/g, "<br>")
              .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>")
              .replace(/- (.*?)(?=\n|$)/g, "• $1");
          }
        } catch (e) {
          console.error('[Chatbox] markdown render error', e);
          contentHtml = (raw || "").replace(/\n/g, "<br>");
        }
      }

      return `
        <div class="${rowClass}">
          ${isBot ? avatar : ""}
          <div class="${bubbleClass}">${contentHtml}</div>
          ${!isBot ? avatar : ""}
        </div>
      `;
    }).join('');

    // Auto scroll xuống đáy
    container.scrollTop = container.scrollHeight;
  }
}

const chatbox = new Chatbox();
