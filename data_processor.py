import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

class FinancialDataProcessor:
    def __init__(self):
        self.data_cache = {}
        
    def load_financial_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        try:
            if file_type.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_type.lower() in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            df = self.clean_data(df)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert date columns
        date_columns = ['date', 'reporting_date', 'period_end']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def perform_financial_analysis(self, df: pd.DataFrame, query_type: str) -> Dict[str, Any]:
        results = {}
        
        if query_type == 'revenue_analysis':
            results = self._revenue_analysis(df)
        elif query_type == 'profitability_analysis':
            results = self._profitability_analysis(df)
        
        return results
    
    def _revenue_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income'])
        if not revenue_col:
            return {'error': 'Revenue column not found'}
        
        return {
            'total_revenue': float(df[revenue_col].sum()),
            'average_revenue': float(df[revenue_col].mean()),
            'median_revenue': float(df[revenue_col].median()),
        }
    
    def _profitability_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        revenue_col = self._find_column(df, ['revenue', 'sales', 'income'])
        cost_col = self._find_column(df, ['cost', 'expense', 'expenditure'])
        
        if not revenue_col or not cost_col:
            return {'error': 'Revenue or cost column not found'}
        
        df['profit'] = df[revenue_col] - df[cost_col]
        df['profit_margin'] = (df['profit'] / df[revenue_col]) * 100
        
        return {
            'total_profit': float(df['profit'].sum()),
            'average_profit_margin': float(df['profit_margin'].mean()),
        }
    
    def _find_column(self, df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
        for name in possible_names:
            if name in df.columns:
                return name
        return None