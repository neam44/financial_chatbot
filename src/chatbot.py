import os
from typing import Dict, List, Any, Optional
import pandas as pd

from data_processor import FinancialDataProcessor
from vector_store import FinancialVectorStore

class FinancialChatbot:
    def __init__(self, api_key: Optional[str] = None):
        self.data_processor = FinancialDataProcessor()
        self.vector_store = FinancialVectorStore()
        self.conversation_history = []
        
        self.analysis_patterns = {
            'revenue': ['revenue', 'sales', 'income', 'earnings'],
            'costs': ['cost', 'expense', 'expenditure', 'spending'],
            'profit': ['profit', 'margin', 'profitability', 'net income'],
            'trend': ['trend', 'growth', 'change', 'over time'],
        }
    
    def process_query(self, user_query: str, financial_data: Optional[pd.DataFrame] = None) -> str:
        intent = self._classify_query_intent(user_query)
        
        analysis_results = None
        if financial_data is not None and not financial_data.empty:
            analysis_type = f"{intent}_analysis"
            analysis_results = self.data_processor.perform_financial_analysis(
                financial_data, analysis_type
            )
        
        response = self._generate_response(user_query, analysis_results, intent)
        
        self.conversation_history.append({
            'user': user_query,
            'assistant': response,
            'intent': intent,
            'analysis': analysis_results
        })
        
        return response
    
    def _classify_query_intent(self, query: str) -> str:
        query_lower = query.lower()
        
        for intent, keywords in self.analysis_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def _generate_response(self, query: str, analysis_results: Dict, intent: str) -> str:
        if not analysis_results or 'error' in analysis_results:
            return f"""I understand you're asking about {intent} analysis. Please upload your financial data (CSV or Excel file) so I can provide specific insights.

I can analyze:
📈 Revenue trends and patterns
💰 Profitability and margins  
📊 Cost breakdowns
🎯 Performance comparisons"""
        
        if intent == 'revenue' and 'total_revenue' in analysis_results:
            return f"""📈 **Revenue Analysis:**

**Total Revenue:** ${analysis_results['total_revenue']:,.2f}
**Average Revenue:** ${analysis_results['average_revenue']:,.2f}
**Median Revenue:** ${analysis_results['median_revenue']:,.2f}

This analysis was performed using vectorized pandas operations for optimal performance."""
        
        elif intent == 'profit' and 'total_profit' in analysis_results:
            return f"""💰 **Profitability Analysis:**

**Total Profit:** ${analysis_results['total_profit']:,.2f}
**Average Profit Margin:** {analysis_results['average_profit_margin']:.2f}%

{"Strong profitability" if analysis_results['average_profit_margin'] > 15 else "Consider margin improvement opportunities"}."""
        
        return f"I've analyzed your {intent} query. The data shows various metrics that can help inform your financial decisions."