import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Paper, Typography } from '@mui/material';

interface VisualizationProps {
    plotData: string | null;
    data: any | null;
}

const Visualization: React.FC<VisualizationProps> = ({ plotData, data }) => {
    if (!plotData && !data) {
        console.log('No data to visualize');
        return null;
    }

    let plotlyData;
    try {
        if (plotData) {
            console.log('Raw plotData:', plotData);
            // Parse the plot data
            plotlyData = JSON.parse(plotData);
            console.log('Parsed plotlyData:', plotlyData);
            
            // Log the structure of the data
            if (plotlyData.data) {
                console.log('Plotly data array:', plotlyData.data);
            }
            if (plotlyData.layout) {
                console.log('Plotly layout:', plotlyData.layout);
            }
        }
    } catch (error) {
        console.error('Error parsing plot data:', error);
    }

    return (
        <Box sx={{ width: '100%', mt: 2 }}>
            {plotlyData && (
                <Paper sx={{ p: 2, mb: 2 }}>
                    <Plot
                        data={plotlyData.data}
                        layout={{
                            ...plotlyData.layout,
                            autosize: true,
                            height: 400,
                            margin: { t: 25, r: 0, l: 30, b: 30 },
                            showlegend: true,
                        }}
                        useResizeHandler={true}
                        style={{ width: '100%', height: '100%' }}
                        config={{
                            responsive: true,
                            displayModeBar: true,
                        }}
                        onError={(error) => {
                            console.error('Plotly render error:', error);
                        }}
                    />
                </Paper>
            )}
            {data && (
                <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                        Data
                    </Typography>
                    <pre style={{ overflow: 'auto', maxHeight: '300px' }}>
                        {JSON.stringify(data, null, 2)}
                    </pre>
                </Paper>
            )}
        </Box>
    );
};

export default Visualization; 