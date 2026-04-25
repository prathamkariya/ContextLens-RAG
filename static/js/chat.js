// static/js/chat.js
document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatHistory = document.getElementById('chat-history');
    const uploadBtn = document.getElementById('upload-btn');
    const sendBtn = document.getElementById('send-btn');

    // Handle PDF Upload
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('pdf-upload');
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('pdf_file', file);

        uploadStatus.textContent = ' Building Vector Index...';
        uploadStatus.className = '';
        uploadBtn.classList.add('loading');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                uploadStatus.textContent = data.message;
                uploadStatus.className = 'success';
            } else {
                uploadStatus.textContent = `Error: ${data.error}`;
                uploadStatus.className = 'error';
            }
        } catch (error) {
            uploadStatus.textContent = 'Upload failed.';
            uploadStatus.className = 'error';
        } finally {
            uploadBtn.classList.remove('loading');
        }
    });

    // Handle Chat Message
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const query = chatInput.value.trim();
        if (!query) return;

        // Append User Message
        appendMessage('user', query);
        chatInput.value = '';
        
        // Disable input while waiting
        chatInput.disabled = true;
        sendBtn.classList.add('loading');

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            
            const data = await response.json();

            if (response.ok) {
                appendBotMessage(data.answer, data.sources);
            } else {
                appendMessage('system', `Error: ${data.error}`);
            }
        } catch (error) {
            appendMessage('system', 'Failed to connect to the reasoning engine.');
        } finally {
            chatInput.disabled = false;
            sendBtn.classList.remove('loading');
            chatInput.focus();
        }
    });

    function appendMessage(sender, text) {
        const div = document.createElement('div');
        div.classList.add('message', `${sender}-message`);
        div.textContent = text;
        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function appendBotMessage(answer, sources) {
        const div = document.createElement('div');
        div.classList.add('message', 'bot-message');
        
        // Add Answer Text
        const textDiv = document.createElement('div');
        textDiv.textContent = answer;
        div.appendChild(textDiv);

        // Add Sources Expander
        if (sources && sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.classList.add('sources-container');

            const toggleBtn = document.createElement('button');
            toggleBtn.classList.add('sources-btn');
            toggleBtn.textContent = 'View Retrieved Evidence (Bridge to Truth)';
            
            const contentDiv = document.createElement('div');
            contentDiv.classList.add('sources-content');
            
            sources.forEach((source, index) => {
                contentDiv.textContent += `[Chunk ${index + 1}]:\n${source}\n\n`;
            });

            toggleBtn.addEventListener('click', () => {
                const isHidden = contentDiv.style.display === 'none' || contentDiv.style.display === '';
                contentDiv.style.display = isHidden ? 'block' : 'none';
            });

            sourcesContainer.appendChild(toggleBtn);
            sourcesContainer.appendChild(contentDiv);
            div.appendChild(sourcesContainer);
        }

        chatHistory.appendChild(div);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});