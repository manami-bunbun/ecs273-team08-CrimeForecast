import React, { useState, useEffect } from 'react';
import { Card, Typography, List, Space, Spin } from 'antd';
import { SafetyOutlined, InfoCircleOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const API_BASE_URL = 'http://localhost:8001';

export default function AdvicePanel({ endDate }) {
  const [advice, setAdvice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAdvice = async () => {
      if (!endDate) return;
      
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(
          `${API_BASE_URL}/api/analysis?end_date=${endDate}`
        );
        
        if (!response.ok) {
          throw new Error('Failed to fetch analysis');
        }
        
        const data = await response.json();
        setAdvice(data.llm_analysis); 
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchAdvice();
  }, [endDate]);

  if (loading) {
    return (
      <Card>
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Spin size="large" />
          <Text>Analyzing crime trends...</Text>
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <Text type="danger">{error}</Text>
      </Card>
    );
  }

  if (!advice) {
    return (
      <Card>
        <Text type="secondary">Select a date to view safety analysis.</Text>
      </Card>
    );
  }

  return (
    <Card>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>

        <div>
          <Title level={5}>
            <InfoCircleOutlined /> Trend Summary
          </Title>
          <Paragraph>{advice.trend_summary}</Paragraph>
        </div>

        <div>
          <Title level={5}>
            <InfoCircleOutlined /> Safety Recommendations
          </Title>
          <List
            size="small"
            dataSource={advice.safety_recommendations}
            renderItem={(item) => (
              <List.Item>
                <Text>{item}</Text>
              </List.Item>
            )}
          />
        </div>

        {/* {advice.confidence_score && (
          <Text type="secondary">
            Analysis confidence: {Math.round(advice.confidence_score * 100)}%
          </Text>
        )} */}
      </Space>
    </Card>
  );
}
