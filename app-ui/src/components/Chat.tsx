import React, { useState, useRef, useEffect } from 'react';
import {
    Box,
    Paper,
    TextField,
    IconButton,
    Typography,
    CircularProgress,
    Alert,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { Message, AnalysisResult } from '../types';
import Visualization from './Visualization';

const API_BASE_URL = 'http://localhost:8000/api/v1';

const Chat: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [currentFile, setCurrentFile] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        setIsLoading(true);
        setError(null);

        try {
            console.log('Uploading file to:', `${API_BASE_URL}/data/upload`);
            const response = await fetch(`${API_BASE_URL}/data/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Upload response error:', errorData);
                throw new Error(errorData.detail || 'Upload failed');
            }

            const data = await response.json();
            console.log('Upload response:', data);
            setCurrentFile(data.filename);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `File "${file.name}" uploaded successfully. You can now ask questions about the data.`,
                result: null
            }]);
        } catch (error) {
            console.error('Upload error:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Failed to upload file: ${error instanceof Error ? error.message : 'Please try again.'}`,
                result: null
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSend = async () => {
        if (!input.trim() || !currentFile) return;

        const userMessage: Message = {
            role: 'user',
            content: input.trim(),
            result: null
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            console.log('Sending query to:', `${API_BASE_URL}/query/process`);
            const response = await fetch(`${API_BASE_URL}/query/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: input.trim(),
                    filename: currentFile
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('Query response error:', errorData);
                throw new Error(errorData.detail || 'Failed to process query');
            }

            const data = await response.json();
            console.log('Query response:', data);

            if (data.success) {
                console.log('Success response data:', data.result);
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: 'Here\'s the analysis of your data:',
                    result: data.result
                }]);
            } else {
                console.error('Error response:', data.error);
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `Error: ${data.error}`,
                    result: null
                }]);
            }
        } catch (error) {
            console.error('Query error:', error);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `Failed to process query: ${error instanceof Error ? error.message : 'Please try again.'}`,
                result: null
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const renderMessage = (message: Message) => {
        if (message.role === 'user') {
            return (
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
                    <Paper
                        sx={{
                            p: 2,
                            backgroundColor: 'primary.main',
                            color: 'white',
                            maxWidth: '70%',
                        }}
                    >
                        <Typography>{message.content}</Typography>
                    </Paper>
                </Box>
            );
        }

        return (
            <Box sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
                <Paper
                    sx={{
                        p: 2,
                        backgroundColor: 'grey.100',
                        maxWidth: '70%',
                    }}
                >
                    <Typography>{message.content}</Typography>
                    {message.result && (
                        <Visualization
                            plotData={message.result.plot}
                            data={message.result.data}
                        />
                    )}
                </Paper>
            </Box>
        );
    };

    return (
        <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                    Data Analysis Chat
                </Typography>
                <input
                    accept=".csv,.xlsx,.xls"
                    style={{ display: 'none' }}
                    id="file-upload"
                    type="file"
                    onChange={handleFileUpload}
                />
                <label htmlFor="file-upload">
                    <IconButton component="span" color="primary">
                        <UploadFileIcon />
                    </IconButton>
                    <Typography component="span" variant="body2">
                        {currentFile ? `Current file: ${currentFile}` : 'Upload a file to begin'}
                    </Typography>
                </label>
            </Paper>

            <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
                {messages.map((message, index) => {
                    console.log('Rendering message:', message);
                    return (
                        <Box
                            key={index}
                            sx={{
                                display: 'flex',
                                justifyContent: message.role === 'user' ? 'flex-end' : 'flex-start',
                                mb: 2,
                            }}
                        >
                            <Paper
                                sx={{
                                    p: 2,
                                    maxWidth: '70%',
                                    backgroundColor: message.role === 'user' ? 'primary.light' : 'background.paper',
                                }}
                            >
                                <Typography variant="body1">{message.content}</Typography>
                                {message.result && (
                                    <Visualization
                                        plotData={message.result.plot}
                                        data={message.result.data}
                                    />
                                )}
                            </Paper>
                        </Box>
                    );
                })}
                <div ref={messagesEndRef} />
            </Box>

            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                    <TextField
                        fullWidth
                        variant="outlined"
                        placeholder="Ask a question about your data..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                        disabled={!currentFile || isLoading}
                    />
                    <IconButton
                        color="primary"
                        onClick={handleSend}
                        disabled={!currentFile || isLoading || !input.trim()}
                    >
                        {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
                    </IconButton>
                </Box>
            </Box>
        </Box>
    );
};

export default Chat; 