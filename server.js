const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();
const PORT = 3000;

// API proxy to forward requests to the backend
app.use('/api', createProxyMiddleware({
  target: 'http://localhost:8000',
  changeOrigin: true,
  pathRewrite: {
    '^/api': '/' // Remove /api prefix when forwarding
  },
  onProxyRes: function(proxyRes, req, res) {
    // Log proxy activity
    console.log(`Proxied ${req.method} ${req.path} -> ${proxyRes.statusCode}`);
  }
}));

// Serve static files
app.use(express.static(__dirname));

// Send all other requests to index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Frontend server running at http://localhost:${PORT}`);
  console.log(`API requests will be proxied to http://localhost:8000`);
}); 
