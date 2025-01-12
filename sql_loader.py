# sql_loader.py

import pandas as pd
from sqlalchemy import create_engine, inspect
import os

class SQLLoader:
    def __init__(self, db_path='sqlite:///bank_transactions.db'):
        self.db_path = db_path
        try:
            self.engine = create_engine(db_path)
            print(f"Connected to database: {db_path}")
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            self.engine = None
    
    def export_to_sql(self, df, table_name):
        try:
            if self.engine:
                df.to_sql(table_name, self.engine, if_exists='replace', index=False)
                print(f"Successfully exported to table: {table_name}")
                return True
            return False
        except Exception as e:
            print(f"Error exporting to SQL table {table_name}: {str(e)}")
            return False
    
    def load_dataframes_to_sql(self, income_df, expenses_df, full_df):
        success = True
        success &= self.export_to_sql(income_df, 'income_transactions')
        success &= self.export_to_sql(expenses_df, 'expense_transactions')
        success &= self.export_to_sql(full_df, 'all_transactions')
        return success
    
    def get_table_names(self):
        if self.engine:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        return []
    
    def read_table(self, table_name):
        try:
            if self.engine and table_name in self.get_table_names():
                return pd.read_sql_table(table_name, self.engine)
            return None
        except Exception as e:
            print(f"Error reading table {table_name}: {str(e)}")
            return None
    
    def get_database_info(self):
        tables = self.get_table_names()
        info = {}
        for table in tables:
            df = self.read_table(table)
            if df is not None:
                info[table] = {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'size': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
                }
        return info
    
    def delete_database(self):
        try:
            if self.engine:
                self.engine.dispose()
            db_file = self.db_path.replace('sqlite:///', '')
            if os.path.exists(db_file):
                os.remove(db_file)
                print("Database deleted successfully")
                return True
            return False
        except Exception as e:
            print(f"Error deleting database: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    loader = SQLLoader()
    
    # Check if tables exist
    print("\nExisting tables:", loader.get_table_names())
    
    # Get database info
    info = loader.get_database_info()
    if info:
        print("\nDatabase Information:")
        for table, details in info.items():
            print(f"\nTable: {table}")
            print(f"Rows: {details['rows']}")
            print(f"Columns: {details['columns']}")
            print(f"Size: {details['size']}")
