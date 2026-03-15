from typing import List, Dict, Union
from crewai.tools import BaseTool
from sqlalchemy import text
from .db_client import engine
from datetime import datetime
import re

class SQLExecutionTool(BaseTool):
    """Tool to safely execute SQL queries with validation and safety checks"""
    name: str = "SQLExecutor"
    description: str = "Safely executes SQL queries with validation and returns results in a readable format"
    
    def _validate_query(self, sql_query: str) -> Dict[str, Union[bool, str]]:
        """Validate SQL query before execution - safety first!"""
        try:
            sql_lower = sql_query.lower().strip()
            
            # Allow CTEs (WITH clauses) and SELECT queries
            if not (sql_lower.startswith('select') or sql_lower.startswith('with')):
                return {
                    "valid": False,
                    "error": "❌ Only SELECT queries and CTEs (WITH clauses) are allowed. This query appears to be: " + sql_lower[:50]
                }
            
            # Safety checks - block dangerous operations
            dangerous_patterns = [
                r'\bdrop\b',
                r'\bdelete\b',
                r'\btruncate\b',
                r'\binsert\b',
                r'\bupdate\b',
                r'\balter\b',
                r'\bcreate\b',
                r'\bexec\b',
                r'\bexecute\b',
                r'\bdeclare\b',
                r'\b@',
                r';\s*--',
                r'--\s*$'
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, sql_lower):
                    return {
                        "valid": False,
                        "error": f"⚠️ SECURITY ALERT: Query contains potentially dangerous operation: '{pattern}'. Only SELECT queries are allowed."
                    }
            
            # Basic validation passed
            return {"valid": True, "message": "✅ Query validated successfully"}
        except Exception as e:
            return {"valid": False, "error": f"❌ Validation error: {str(e)}"}
    
    def _format_results(self, results: List[Dict], query: str) -> str:
        """Format query results in a readable, business-friendly format"""
        if not results:
            return "🔍 Query executed successfully but returned no results."
        
        # Get column names
        columns = list(results[0].keys())
        
        # Create a clean table format
        table_header = " | ".join([f"{col:<20}" for col in columns])
        separator = "-" * len(table_header)
        
        # Format rows
        rows = []
        for row in results:
            formatted_row = " | ".join([f"{str(row[col]):<20}" for col in columns])
            rows.append(formatted_row)
        
        # Add summary statistics
        row_count = len(results)
        summary = f"\n📊 Query Results Summary:"
        summary += f"\n   • Rows returned: {row_count}"
        summary += f"\n   • Columns: {', '.join(columns)}"
        summary += f"\n   • Query executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return f"""
🔍 QUERY EXECUTED:
{query}

📊 RESULTS:
{table_header}
{separator}
{chr(10).join(rows)}

{summary}
"""
    
    def _run(self, sql_query: str) -> str:
        """Execute SQL query safely with validation"""
        try:
            # Step 1: Validate the query
            validation = self._validate_query(sql_query)
            if not validation["valid"]:
                return validation["error"]
            
            # Step 2: Execute the query
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                
                if not rows:
                    return "✅ Query executed successfully but returned no results."
                
                # Convert to dictionary format
                columns = result.keys()
                results = [dict(zip(columns, row)) for row in rows]
                
                # Step 3: Format results beautifully
                return self._format_results(results, sql_query)
                
        except Exception as e:
            error_msg = str(e)
            # Handle common SQL errors gracefully
            if "invalid column name" in error_msg.lower():
                return f"❌ COLUMN ERROR: One or more columns in your query don't exist. Check the column names against the schema."
            elif "invalid object name" in error_msg.lower():
                return f"❌ TABLE ERROR: The table name is incorrect. Use fully qualified names like [DATA].[dbo].[TABLE_NAME]."
            elif "conversion failed" in error_msg.lower():
                return f"❌ DATA TYPE ERROR: There's a data type conversion issue. Check your WHERE clause conditions."
            else:
                return f"❌ EXECUTION ERROR: {error_msg}"
