// static/js/chat.js
// Fixes: #16 char limit, #17 typing indicator, #18 scroll button,
//        #19 source badge, #20 copy button, #22 filename display, #23 re-upload confirm,
//        #21 Shift+Enter for newline / Enter to submit
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm     = document.getElementById('upload-form');
    const uploadStatus   = document.getElementById('upload-status');
    const fileInput      = document.getElementById('pdf-upload');
    const fileLabel      = document.getElementById('file-label');
    const chatForm       = document.getElementById('chat-form');
    const chatInput      = document.getElementById('chat-input');
    const charCounter    = document.getElementById('char-counter');
    const chatHistory    = document.getElementById('chat-history');
    const uploadBtn      = document.getElementById('upload-btn');
    const sendBtn        = document.getElementById('send-btn');
    const scrollDownBtn  = document.getElementById('scroll-down-btn');

    const MAX_CHARS = 500;
    let documentLoaded = false;   // tracks whether a PDF is indexed

    // ── #22: Show selected filename in a styled label ─────────────────────────
    fileInput.addEventListener('change', () => {
        const name = fileInput.files[0]?.name ?? 'No file chosen';
        fileLabel.textContent = name;
        fileLabel.title = name;
    });

    // ── #16: Character counter ────────────────────────────────────────────────
    chatInput.addEventListener('input', () => {
        const remaining = MAX_CHARS - chatInput.value.length;
        charCounter.textContent = `${chatInput.value.length} / ${MAX_CHARS}`;
        charCounter.classList.toggle('counter-warn', remaining < 50);
        charCounter.classList.toggle('counter-error', remaining < 0);
        sendBtn.disabled = chatInput.value.length > MAX_CHARS;
    });

    // ── #18: Scroll-to-bottom button (appears when user scrolls up) ──────────
    chatHistory.addEventListener('scroll', () => {
        const atBottom =
            chatHistory.scrollHeight - chatHistory.scrollTop - chatHistory.clientHeight < 80;
        scrollDownBtn.classList.toggle('visible', !atBottom);
    });

    scrollDownBtn.addEventListener('click', () => {
        chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    });

    // ── PDF Upload ────────────────────────────────────────────────────────────
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;

        // #23: Warn before destroying existing conversation context
        if (documentLoaded) {
            const ok = confirm(
                'Uploading a new document will reset your current conversation. Continue?'
            );
            if (!ok) return;
        }

        const formData = new FormData();
        formData.append('pdf_file', file);

        uploadStatus.textContent = 'Building index…';
        uploadStatus.className = '';
        uploadBtn.disabled = true;
        uploadBtn.classList.add('loading');

        try {
            const res  = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();

            if (res.ok) {
                uploadStatus.textContent = '✓ ' + data.message;
                uploadStatus.className = 'success';
                documentLoaded = true;
                // Clear old messages on successful new upload (#24)
                clearChat();
                appendSystemMessage('New document indexed. Ask your first question below.');
            } else {
                uploadStatus.textContent = `✗ ${data.error}`;
                uploadStatus.className = 'error';
            }
        } catch {
            uploadStatus.textContent = '✗ Upload failed. Check your connection.';
            uploadStatus.className = 'error';
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.classList.remove('loading');
        }
    });

    // ── #21: Shift+Enter = newline, Enter alone = submit ────────────────────
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!chatInput.disabled && chatInput.value.trim().length > 0) {
                chatForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
            }
        }
    });

    // ── Chat ──────────────────────────────────────────────────────────────────
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query || query.length > MAX_CHARS) return;

        appendUserMessage(query);
        chatInput.value = '';
        charCounter.textContent = `0 / ${MAX_CHARS}`;

        // #17: Typing indicator bubble in chat history
        const typingEl = appendTypingIndicator();

        chatInput.disabled = true;
        sendBtn.disabled = true;
        sendBtn.classList.add('loading');

        try {
            const res  = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });
            const data = await res.json();

            typingEl.remove();

            if (res.ok) {
                appendBotMessage(data.answer, data.sources);
            } else {
                appendSystemMessage(`Error: ${data.error}`);
            }
        } catch {
            typingEl.remove();
            appendSystemMessage('Failed to connect to the reasoning engine.');
        } finally {
            chatInput.disabled = false;
            sendBtn.disabled = false;
            sendBtn.classList.remove('loading');
            chatInput.focus();
        }
    });

    // ── Render helpers ────────────────────────────────────────────────────────

    function clearChat() {
        chatHistory.innerHTML = '';
    }

    function scrollToBottom() {
        chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
    }

    function appendSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'message system-message';
        div.textContent = text;
        chatHistory.appendChild(div);
        scrollToBottom();
    }

    function appendUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message user-message';
        div.textContent = text;
        chatHistory.appendChild(div);
        scrollToBottom();
    }

    // #17: Three-dot animated typing indicator
    function appendTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'message bot-message typing-indicator';
        div.innerHTML = '<span></span><span></span><span></span>';
        chatHistory.appendChild(div);
        scrollToBottom();
        return div;
    }

    function appendBotMessage(answer, sources) {
        const div = document.createElement('div');
        div.className = 'message bot-message';

        // Answer text with markdown rendering
        const textDiv = document.createElement('div');
        textDiv.className = 'answer-text';
        if (typeof marked !== 'undefined') {
            textDiv.innerHTML = marked.parse(answer);
        } else {
            textDiv.textContent = answer;
        }
        div.appendChild(textDiv);

        // #20: Copy button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.title = 'Copy answer';
        copyBtn.innerHTML = '📋 Copy';
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(answer).then(() => {
                copyBtn.innerHTML = '✓ Copied';
                setTimeout(() => { copyBtn.innerHTML = '📋 Copy'; }, 2000);
            });
        });
        div.appendChild(copyBtn);

        // Sources section — #9: Now shows page number and filename
        if (sources && sources.length > 0) {
            // #19: Inline badge showing chunk count
            const badge = document.createElement('span');
            badge.className = 'source-badge';
            badge.textContent = `📎 ${sources.length} source${sources.length > 1 ? 's' : ''}`;
            div.insertBefore(badge, textDiv);   // badge before answer text

            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';

            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'sources-btn';
            toggleBtn.textContent = `View Retrieved Evidence (${sources.length} chunk${sources.length > 1 ? 's' : ''})`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'sources-content';
            contentDiv.style.display = 'none';

            // BUG-010 fix carried forward: one <pre> per chunk
            sources.forEach((src, i) => {
                const chunkEl = document.createElement('pre');
                chunkEl.className = 'source-chunk';
                // #9: Show page and filename from metadata
                const label = `[Chunk ${i + 1}] — Page ${src.page ?? '?'} · ${src.source ?? 'document'}`;
                chunkEl.textContent = `${label}\n${src.text ?? src}`;
                contentDiv.appendChild(chunkEl);
            });

            toggleBtn.addEventListener('click', () => {
                const hidden = contentDiv.style.display === 'none';
                contentDiv.style.display = hidden ? 'block' : 'none';
                toggleBtn.textContent = hidden
                    ? `Hide Evidence (${sources.length} chunk${sources.length > 1 ? 's' : ''})`
                    : `View Retrieved Evidence (${sources.length} chunk${sources.length > 1 ? 's' : ''})`;
            });

            sourcesContainer.appendChild(toggleBtn);
            sourcesContainer.appendChild(contentDiv);
            div.appendChild(sourcesContainer);
        }

        chatHistory.appendChild(div);
        scrollToBottom();
    }
});