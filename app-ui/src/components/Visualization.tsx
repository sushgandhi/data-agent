import React from 'react';
import Plot from 'react-plotly.js';
import { Box, Paper, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Grid } from '@mui/material';

interface VisualizationProps {
    plotData: string | null;
    data: {
        shape: [number, number];
        columns: string[];
        dtypes: Record<string, string>;
        missing_values: Record<string, number>;
        data: any;
    } | null;
}

const Visualization: React.FC<VisualizationProps> = ({ plotData, data }) => {
    // Helper function to convert any value to a string representation
    const valueToString = (value: any): string => {
        if (value === null || value === undefined) return '';
        if (typeof value === 'object') {
            return JSON.stringify(value);
        }
        return String(value);
    };

    // Convert data.data to array if it's not already
    const dataArray = React.useMemo(() => {
        if (!data?.data) return [];
        
        if (Array.isArray(data.data)) {
            // Ensure each row is an array
            return data.data.map(row => Array.isArray(row) ? row : [row]);
        } else if (typeof data.data === 'object' && data.data !== null) {
            // If data is an object, convert it to array of arrays
            const rows = [];
            for (let i = 0; i < Math.min(5, data.shape[0]); i++) {
                const row = data.columns.map(col => {
                    const value = data.data[col];
                    if (Array.isArray(value)) {
                        return value[i];
                    }
                    return value;
                });
                rows.push(row);
            }
            return rows;
        }
        return [];
    }, [data]);

    if (!plotData && !data) {
        console.log('No data to visualize');
        return null;
    }

    let plotlyData;
    try {
        if (plotData) {
            console.log('Raw plotData:', plotData);
            plotlyData = JSON.parse(plotData);
            console.log('Parsed plotlyData:', plotlyData);
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
                <Grid container spacing={2}>
                    <Grid item xs={12}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Data Summary
                            </Typography>
                            <Grid container spacing={2}>
                                <Grid item xs={12} sm={6}>
                                    <Typography variant="subtitle1">
                                        Shape: {data.shape[0]} rows × {data.shape[1]} columns
                                    </Typography>
                                </Grid>
                                <Grid item xs={12} sm={6}>
                                    <Typography variant="subtitle1">
                                        Total Columns: {data.columns.length}
                                    </Typography>
                                </Grid>
                            </Grid>
                        </Paper>
                    </Grid>
                    <Grid item xs={12}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Column Information
                            </Typography>
                            <TableContainer>
                                <Table size="small">
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Column</TableCell>
                                            <TableCell>Data Type</TableCell>
                                            <TableCell>Missing Values</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {data.columns.map((column) => (
                                            <TableRow key={column}>
                                                <TableCell>{column}</TableCell>
                                                <TableCell>{data.dtypes[column]}</TableCell>
                                                <TableCell>{data.missing_values[column]}</TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Paper>
                    </Grid>
                    <Grid item xs={12}>
                        <Paper sx={{ p: 2 }}>
                            <Typography variant="h6" gutterBottom>
                                Data Preview
                            </Typography>
                            <TableContainer sx={{ maxHeight: 400 }}>
                                <Table size="small" stickyHeader>
                                    <TableHead>
                                        <TableRow>
                                            {data.columns.map((column) => (
                                                <TableCell key={column}>{column}</TableCell>
                                            ))}
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {dataArray.map((row: any[], index: number) => (
                                            <TableRow key={index}>
                                                {Array.isArray(row) ? row.map((cell, cellIndex) => (
                                                    <TableCell key={cellIndex}>{valueToString(cell)}</TableCell>
                                                )) : (
                                                    <TableCell>{valueToString(row)}</TableCell>
                                                )}
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </Paper>
                    </Grid>
                </Grid>
            )}
        </Box>
    );
};

export default Visualization; 