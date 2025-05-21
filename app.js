// Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINTS = {
    query: `${API_BASE_URL}/query`,
    metadata: `${API_BASE_URL}/metadata`,
    health: `${API_BASE_URL}/health`,
    upload: `${API_BASE_URL}/upload`
};

// DOM Elements
const fileUploadInput = document.getElementById('file-upload');
const fileName = document.getElementById('file-name');
const uploadBtn = document.getElementById('upload-btn');
const currentFileDisplay = document.getElementById('current-file');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatForm = document.getElementById('chat-form');
const sendBtn = document.querySelector('.send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const chatHistoryList = document.getElementById('chat-history-list');

// Templates
const userMessageTemplate = document.getElementById('user-message-template');
const botMessageTemplate = document.getElementById('bot-message-template');
const botMessageWithTableTemplate = document.getElementById('bot-message-with-table-template');
const botMessageWithImageTemplate = document.getElementById('bot-message-with-image-template');
const loadingTemplate = document.getElementById('loading-template');
const errorMessageTemplate = document.getElementById('error-message-template');
const chatHistoryItemTemplate = document.getElementById('chat-history-item-template');

// State
let currentState = {
    filePath: null,
    fileName: null,
    chatHistory: [],
    currentChatId: generateId(),
    chatSessions: {}
};

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

fileUploadInput.addEventListener('change', handleFileSelection);
uploadBtn.addEventListener('click', handleFileUpload);
chatForm.addEventListener('submit', handleChatSubmit);
newChatBtn.addEventListener('click', startNewChat);

// Auto-resize textarea
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = (chatInput.scrollHeight) + 'px';
});

// Initialize the application
function initializeApp() {
    // Check API health
    checkApiHealth()
        .then(healthy => {
            if (!healthy) {
                showErrorMessage('API server is not available. Please make sure the server is running at ' + API_BASE_URL);
            }
        });

    // Load chat history from localStorage
    loadState();
    
    // Render chat history
    renderChatHistory();
}

// Check API health
async function checkApiHealth() {
    try {
        const response = await fetch(API_ENDPOINTS.health);
        const data = await response.json();
        return data.status === 'healthy';
    } catch (error) {
        console.error('API health check failed:', error);
        return false;
    }
}

// Handle file selection
function handleFileSelection(event) {
    const file = event.target.files[0];
    
    if (file) {
        fileName.textContent = file.name;
        uploadBtn.disabled = false;
    } else {
        fileName.textContent = '';
        uploadBtn.disabled = true;
    }
}

// Handle file upload
async function handleFileUpload() {
    const file = fileUploadInput.files[0];
    
    if (!file) {
        showErrorMessage('Please select a file to upload.');
        return;
    }
    
    // Disable upload button while uploading
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    
    // Show loading message
    const loadingElement = addLoadingIndicator();
    
    try {
        // Create FormData object to send the file
        const formData = new FormData();
        formData.append('file', file);
        
        // Send file to server
        const response = await fetch(API_ENDPOINTS.upload, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`File upload failed: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Remove loading indicator
        removeLoadingIndicator(loadingElement);
        
        // Check if upload was successful
        if (data.success) {
            // Store file path from server response
            currentState.filePath = data.file_path;
            currentState.fileName = file.name;
            
            // Update UI
            currentFileDisplay.textContent = `File: ${file.name}`;
            
            // Enable chat input
            chatInput.disabled = false;
            sendBtn.disabled = false;
            
            // Update chat session
            currentState.chatSessions[currentState.currentChatId] = {
                fileName: file.name,
                filePath: data.file_path,
                messages: []
            };
            
            // Save state
            saveState();
            
            // Clear file input
            fileUploadInput.value = '';
            fileName.textContent = '';
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Upload';
            
            // Add chat history item
            addChatHistoryItem(currentState.currentChatId, `Chat about ${file.name}`);
            
            // Show success message
            showBotMessage(`File loaded: ${file.name}. You can now ask questions about your data.`);
        } else {
            throw new Error(data.error || 'File upload failed');
        }
    } catch (error) {
        // Remove loading indicator
        removeLoadingIndicator(loadingElement);
        
        // Reset upload button
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload';
        
        // Show error message
        showErrorMessage(`Error uploading file: ${error.message}`);
    }
}

// Handle chat submission
async function handleChatSubmit(event) {
    event.preventDefault();
    
    const query = chatInput.value.trim();
    
    if (!query) return;
    
    if (!currentState.filePath) {
        showErrorMessage('Please upload a file first.');
        return;
    }
    
    // Add user message
    showUserMessage(query);
    
    // Clear input and reset height
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Disable input while processing
    chatInput.disabled = true;
    sendBtn.disabled = true;
    
    // Add loading indicator
    const loadingElement = addLoadingIndicator();
    
    try {
        const response = await sendQueryToApi(query, currentState.filePath);
        
        // Remove loading indicator
        removeLoadingIndicator(loadingElement);
        
        // Enable input
        chatInput.disabled = false;
        sendBtn.disabled = false;
        
        // Process response
        processApiResponse(response);
        
        // Save message to current chat session
        if (currentState.chatSessions[currentState.currentChatId]) {
            currentState.chatSessions[currentState.currentChatId].messages.push({
                role: 'user',
                content: query
            });
            
            currentState.chatSessions[currentState.currentChatId].messages.push({
                role: 'bot',
                content: response
            });
            
            // Save state
            saveState();
        }
        
        // Focus input
        chatInput.focus();
        
    } catch (error) {
        // Remove loading indicator
        removeLoadingIndicator(loadingElement);
        
        // Enable input
        chatInput.disabled = false;
        sendBtn.disabled = false;
        
        // Show error message
        showErrorMessage(`Error: ${error.message}`);
        
        // Focus input
        chatInput.focus();
    }
}

// Send query to API
async function sendQueryToApi(query, filePath) {
    try {
        const response = await fetch(API_ENDPOINTS.query, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                file_path: filePath
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Process API response
function processApiResponse(response) {
    if (response.type === 'error') {
        showErrorMessage(`Error: ${response.error_message}`);
        return;
    }
    
    if (response.type === 'text') {
        showBotMessage(response.content);
        return;
    }
    
    if (response.type === 'function_call') {
        const result = response.result;
        
        if (result && result.success) {
            const functionResult = result.result;
            
            if (functionResult.message) {
                // Check if we have a DataFrame result
                if (functionResult.result_df && Array.isArray(functionResult.result_df)) {
                    showBotMessageWithTable(functionResult.message, functionResult.result_df);
                }
                // Check if we have a plot URL
                else if (functionResult.plot_url) {
                    // Use the actual plot URL from the response
                    showBotMessageWithImage(functionResult.message, functionResult.plot_url);
                }
                else {
                    showBotMessage(functionResult.message);
                }
            } else {
                showBotMessage('Operation completed successfully.');
            }
        } else {
            showErrorMessage('Function execution failed.');
        }
    }
}

// Show user message
function showUserMessage(message) {
    const messageElement = userMessageTemplate.content.cloneNode(true);
    messageElement.querySelector('p').textContent = message;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Show bot message
function showBotMessage(message) {
    const messageElement = botMessageTemplate.content.cloneNode(true);
    messageElement.querySelector('p').textContent = message;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Show bot message with table
function showBotMessageWithTable(message, data) {
    const messageElement = botMessageWithTableTemplate.content.cloneNode(true);
    messageElement.querySelector('p').textContent = message;
    
    // Create table
    const tableContainer = messageElement.querySelector('.table-container');
    const table = document.createElement('table');
    
    // Create table header
    if (data.length > 0) {
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        for (const key of Object.keys(data[0])) {
            const th = document.createElement('th');
            th.textContent = key;
            headerRow.appendChild(th);
        }
        
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // Create table body
        const tbody = document.createElement('tbody');
        
        for (const row of data) {
            const tr = document.createElement('tr');
            
            for (const key of Object.keys(data[0])) {
                const td = document.createElement('td');
                td.textContent = row[key] !== null ? row[key] : '';
                tr.appendChild(td);
            }
            
            tbody.appendChild(tr);
        }
        
        table.appendChild(tbody);
        tableContainer.appendChild(table);
    }
    
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Show bot message with image
function showBotMessageWithImage(message, imageUrl) {
    const messageElement = botMessageWithImageTemplate.content.cloneNode(true);
    messageElement.querySelector('p').textContent = message;
    messageElement.querySelector('img').src = imageUrl;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Show error message
function showErrorMessage(message) {
    const messageElement = errorMessageTemplate.content.cloneNode(true);
    messageElement.querySelector('p').textContent = message;
    chatMessages.appendChild(messageElement);
    scrollToBottom();
}

// Add loading indicator
function addLoadingIndicator() {
    const loadingElement = loadingTemplate.content.cloneNode(true);
    chatMessages.appendChild(loadingElement);
    scrollToBottom();
    return chatMessages.lastChild;
}

// Remove loading indicator
function removeLoadingIndicator(element) {
    if (element && element.parentNode) {
        element.parentNode.removeChild(element);
    }
}

// Scroll to bottom of chat messages
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Start new chat
function startNewChat() {
    // Generate new chat ID
    currentState.currentChatId = generateId();
    
    // Clear chat messages
    clearChatMessages();
    
    // Reset current file display
    currentFileDisplay.textContent = 'No file loaded';
    
    // Disable chat input
    chatInput.disabled = true;
    sendBtn.disabled = true;
    
    // Clear current file path
    currentState.filePath = null;
    currentState.fileName = null;
    
    // Create new chat session
    currentState.chatSessions[currentState.currentChatId] = {
        fileName: null,
        filePath: null,
        messages: []
    };
    
    // Save state
    saveState();
    
    // Add chat history item
    addChatHistoryItem(currentState.currentChatId, 'New Chat');
}

// Clear chat messages
function clearChatMessages() {
    // Keep only the welcome message
    const welcomeMessage = chatMessages.firstChild;
    chatMessages.innerHTML = '';
    chatMessages.appendChild(welcomeMessage);
}

// Add chat history item
function addChatHistoryItem(chatId, title) {
    const historyItem = chatHistoryItemTemplate.content.cloneNode(true);
    const listItem = historyItem.querySelector('li');
    
    listItem.dataset.chatId = chatId;
    listItem.querySelector('span').textContent = title;
    
    // Add click event
    listItem.addEventListener('click', () => {
        loadChat(chatId);
    });
    
    // Add to chat history
    chatHistoryList.insertBefore(historyItem, chatHistoryList.firstChild);
    
    // Update current state
    currentState.chatHistory.unshift({
        id: chatId,
        title
    });
    
    // Save state
    saveState();
}

// Load chat
function loadChat(chatId) {
    // Check if chat exists
    if (!currentState.chatSessions[chatId]) {
        showErrorMessage('Chat session not found.');
        return;
    }
    
    // Set current chat ID
    currentState.currentChatId = chatId;
    
    // Clear chat messages
    clearChatMessages();
    
    // Set current file display
    const session = currentState.chatSessions[chatId];
    if (session.fileName) {
        currentFileDisplay.textContent = `File: ${session.fileName}`;
        currentState.filePath = session.filePath;
        currentState.fileName = session.fileName;
        
        // Enable chat input
        chatInput.disabled = false;
        sendBtn.disabled = false;
    } else {
        currentFileDisplay.textContent = 'No file loaded';
        currentState.filePath = null;
        currentState.fileName = null;
        
        // Disable chat input
        chatInput.disabled = true;
        sendBtn.disabled = true;
    }
    
    // Load messages
    if (session.messages && session.messages.length > 0) {
        for (const message of session.messages) {
            if (message.role === 'user') {
                showUserMessage(message.content);
            } else if (message.role === 'bot') {
                processApiResponse(message.content);
            }
        }
    }
    
    // Save state
    saveState();
}

// Render chat history
function renderChatHistory() {
    // Clear chat history
    chatHistoryList.innerHTML = '';
    
    // Add chat history items
    for (const chat of currentState.chatHistory) {
        const historyItem = chatHistoryItemTemplate.content.cloneNode(true);
        const listItem = historyItem.querySelector('li');
        
        listItem.dataset.chatId = chat.id;
        listItem.querySelector('span').textContent = chat.title;
        
        // Add click event
        listItem.addEventListener('click', () => {
            loadChat(chat.id);
        });
        
        // Add to chat history
        chatHistoryList.appendChild(historyItem);
    }
}

// Generate unique ID
function generateId() {
    return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
}

// Save state to localStorage
function saveState() {
    localStorage.setItem('dataFlowAgentState', JSON.stringify(currentState));
}

// Load state from localStorage
function loadState() {
    const savedState = localStorage.getItem('dataFlowAgentState');
    
    if (savedState) {
        currentState = JSON.parse(savedState);
    }
} 
