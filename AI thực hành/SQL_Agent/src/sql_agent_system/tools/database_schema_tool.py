from typing import List, Dict, Any
from crewai.tools import BaseTool
from sqlalchemy import text
from .db_client import engine
import json

class DatabaseSchemaTool(BaseTool):
    """Tool to discover all tables and read their actual descriptions from extended properties"""
    name: str = "DatabaseSchemaInspector"
    description: str = "Discovers all tables and reads their actual business descriptions from SQL Server extended properties"
    
    def _get_all_tables(self) -> List[str]:
        """Get all user tables from the database"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE' 
                    AND TABLE_SCHEMA = 'dbo'
                    ORDER BY TABLE_NAME
                """))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            return f"Error getting tables: {str(e)}"
    
    def _get_table_description(self, table_name: str) -> str:
        """Get the actual table description from extended properties"""
        try:
            with engine.connect() as conn:
                # Correct way to get table description in SQL Server
                result = conn.execute(text("""
                    SELECT 
                        CAST(value AS NVARCHAR(MAX)) AS table_description
                    FROM fn_listextendedproperty (
                        NULL, 'Schema', 'dbo',
                        'Table', :table_name,
                        NULL, NULL
                    )
                    WHERE name = 'MS_Description'
                """), {"table_name": table_name}).fetchone()
                
                if result and result[0]:
                    return str(result[0]).strip()
                return "No business description available for this table"
        except Exception as e:
            return f"Error getting table description: {str(e)}"
    
    def _get_column_descriptions(self, table_name: str) -> Dict[str, str]:
        """Get actual column descriptions from extended properties"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        objname AS column_name,
                        CAST(value AS NVARCHAR(MAX)) AS column_description
                    FROM fn_listextendedproperty (
                        NULL, 'Schema', 'dbo',
                        'Table', :table_name,
                        'Column', NULL
                    )
                    WHERE name = 'MS_Description'
                """), {"table_name": table_name})
                
                descriptions = {}
                for row in result.fetchall():
                    descriptions[row[0]] = str(row[1]) if row[1] else "No description available"
                return descriptions
        except Exception as e:
            return {"error": f"Error getting column descriptions: {str(e)}"}
    
    def _get_table_schema(self, table_name: str) -> str:
        """Get complete schema with ACTUAL descriptions including table description"""
        try:
            with engine.connect() as conn:
                # Get column information using INFORMATION_SCHEMA (correct SQL Server syntax)
                result = conn.execute(text(f"""
                    SELECT 
                        COLUMN_NAME,
                        DATA_TYPE,
                        ISNULL(CHARACTER_MAXIMUM_LENGTH, 0) as CHARACTER_MAXIMUM_LENGTH,
                        IS_NULLABLE,
                        ISNULL(COLUMN_DEFAULT, 'No default') as COLUMN_DEFAULT
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = '{table_name}'
                    AND TABLE_SCHEMA = 'dbo'
                    ORDER BY ORDINAL_POSITION
                """))
                
                # Get actual descriptions
                column_descriptions = self._get_column_descriptions(table_name)
                table_description = self._get_table_description(table_name)
                
                columns = []
                for row in result.fetchall():
                    col_name = row[0]
                    data_type = row[1]
                    max_length = row[2]
                    nullable = row[3]
                    default_val = row[4]
                    
                    # Format data type with length if applicable
                    if max_length > 0 and data_type in ['nvarchar', 'varchar', 'char']:
                        data_type = f"{data_type}({max_length})"
                    
                    col_info = f"- {col_name} ({data_type}, nullable: {nullable}"
                    if default_val != 'No default':
                        col_info += f", default: {default_val}"
                    col_info += ")"
                    
                    # Add the ACTUAL business description
                    actual_desc = column_descriptions.get(col_name, "No description available")
                    col_info += f"\n  Business Description: {actual_desc}"
                    
                    columns.append(col_info)
                
                # Get sample data
                try:
                    sample_result = conn.execute(text(f"""
                        SELECT TOP 3 * FROM dbo.{table_name}
                    """))
                    
                    sample_data = []
                    for row in sample_result.fetchall():
                        sample_data.append(dict(row._mapping))
                except Exception as e:
                    sample_data = []
                
                # Use the ACTUAL table description you provided
                schema_info = f"""
                                TABLE: dbo.{table_name}
                                BUSINESS PURPOSE:
                                {table_description}

                                COLUMNS:
                                {chr(10).join(columns)}

                                SAMPLE DATA (first 3 rows):
                                {json.dumps(sample_data, indent=2, default=str) if sample_data else "No sample data available"}
                                """
                return schema_info
                
        except Exception as e:
            return f"Error getting schema for {table_name}: {str(e)}"
    
    def _run(self, instruction: str = "discover_all_tables") -> str:
        """Main function - reads actual metadata including your table descriptions"""
        if instruction == "discover_all_tables":
            tables = self._get_all_tables()
            if isinstance(tables, str):  # Error occurred
                return tables
            
            schema_summary = "🏢 DATABASE SCHEMA FOR DATA DATABASE (USING ACTUAL BUSINESS DESCRIPTIONS):\n\n"
            schema_summary += f"📋 Total tables found: {len(tables)}\n\n"
            
            # Get schema for each table using REAL descriptions
            table_schemas = []
            for table_name in tables:
                schema_info = self._get_table_schema(table_name)
                table_schemas.append(schema_info)
            
            # Create table of contents
            schema_summary += "📖 TABLE OF CONTENTS:\n"
            for i, table_name in enumerate(tables, 1):
                schema_summary += f"{i}. 📊 dbo.{table_name}\n"
            
            schema_summary += "\n" + "="*80 + "\n\n"
            schema_summary += "\n\n".join(table_schemas)
            
            return schema_summary
        
        elif instruction.startswith("get_schema_for:"):
            table_name = instruction.split(":", 1)[1].strip()
            return self._get_table_schema(table_name)
        
        return "Unknown instruction. Use 'discover_all_tables' or 'get_schema_for:<table_name>'"

