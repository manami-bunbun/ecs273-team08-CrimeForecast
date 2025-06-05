import React, { useState, useEffect, useRef } from 'react';
import { Box, List, ListItem, Typography, Divider } from '@mui/material';
import { format } from 'date-fns';

const API_BASE_URL = 'http://localhost:8001';

export default function NewsList({ endDate }) {
    const [articles, setArticles] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const containerRef = useRef(null);

    useEffect(() => {
        if (!endDate) {
            setError("Please select a date range");
            return;
        }
        
        setLoading(true);
        setError(null);

        fetch(`${API_BASE_URL}/api/analysis?end_date=${endDate}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    setError(data.error);
                } else {
                    setArticles(data.news || []);
                }
            })
            .catch(err => {
                console.error("Failed to fetch news:", err);
                setError("Failed to fetch news. Please try again later.");
            })
            .finally(() => {
                setLoading(false);
            });
    }, [endDate]);

    if (loading) {
        return (
            <Box sx={{ p: 2 }}>
                <Typography>Loading news...</Typography>
            </Box>
        );
    }

    if (error) {
        return (
            <Box sx={{ p: 2 }}>
                <Typography color="error">{error}</Typography>
            </Box>
        );
    }

    if (!articles.length) {
        return (
            <Box sx={{ p: 2 }}>
                <Typography>No relevant news found for the selected date range</Typography>
            </Box>
        );
    }

    return (
        <Box ref={containerRef}>
            <List sx={{ 
                width: '100%', 
                bgcolor: 'background.paper',
                '& .MuiListItem-root': {
                    transition: 'background-color 0.2s ease',
                    cursor: 'pointer',
                    '&:hover': {
                        backgroundColor: 'rgba(0, 0, 0, 0.04)',
                    }
                }
            }}>
                {articles.map((item, index) => (
                    <React.Fragment key={index}>
                        <ListItem
                            component="a"
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            sx={{
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'flex-start',
                                py: 2,
                                textDecoration: 'none',
                                color: 'inherit',
                            }}
                        >
                            <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ mb: 1 }}
                            >
                                {format(new Date(item.published_date), 'MMM d, yyyy')}
                            </Typography>
                            <Typography
                                variant="subtitle1"
                                sx={{
                                    color: 'primary.main',
                                    '&:hover': {
                                        textDecoration: 'underline',
                                    },
                                    fontWeight: 500,
                                }}
                            >
                                {item.title}
                            </Typography>
                        </ListItem>
                        {index < articles.length - 1 && (
                            <Divider sx={{ 
                                borderColor: 'rgba(0, 0, 0, 0.08)',
                                margin: '0 16px'
                            }} />
                        )}
                    </React.Fragment>
                ))}
            </List>
        </Box>
    );
}

function readNewsFile(text) {
    const blocks = text.split(/\r?\n\s*\r?\n/);
    const news = [];

    blocks.forEach(b => {
        const lines = b.trim().split(/\r?\n/);
        if (lines.length < 4) return;

        const titleLine = lines.find(line => line.startsWith("Title:"));
        const dateLine = lines.find(line => line.startsWith("Date:"));
        const urlLine = lines.find(line => line.startsWith("URL:"));
        const contentIndex = lines.findIndex(line => line.startsWith("Content:"));

        if (!titleLine || !dateLine || contentIndex === -1) return;

        const title = titleLine.slice(6).trim();
        const dateStr = dateLine.slice(5).trim();
        const url = urlLine ? urlLine.slice(4).trim() : null;
        const content = lines.slice(contentIndex + 1).join("\n").trim();

        news.push({
            title,
            date: new Date(dateStr),
            url,
            content,
        });
    });

    return news;
}

