export interface Message {
    role: 'user' | 'assistant';
    content: string;
    result: AnalysisResult | null;
}

export interface AnalysisResult {
    plot: string | null;
    data: any | null;
}

export interface ChatState {
    messages: Message[];
    currentFile: string | null;
    isLoading: boolean;
    error: string | null;
} 