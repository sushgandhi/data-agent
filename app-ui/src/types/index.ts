export interface Message {
    role: 'user' | 'assistant';
    content: string;
    result: AnalysisResult | null;
}

export interface AnalysisResult {
    plot: string | null;
    data: {
        shape: [number, number];
        columns: string[];
        dtypes: Record<string, string>;
        missing_values: Record<string, number>;
        data: any;
    } | null;
}

export interface ChatState {
    messages: Message[];
    currentFile: string | null;
    isLoading: boolean;
    error: string | null;
} 