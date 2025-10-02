import os
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables
load_dotenv()

# Get database URL from environment
DB_URL = os.getenv('DB_URL')
if not DB_URL:
    raise ValueError("DB_URL environment variable is not set. Please check your .env file.")

# Create engine
engine = create_engine(DB_URL)

def get_enum_values(conn, table_name, column_name):
    """Get possible values for an enum column"""
    try:
        result = conn.execute(
            text("""
                SELECT column_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = :table_name 
                AND column_name = :column_name;
            """),
            {"table_name": table_name, "column_name": column_name}
        )
        enum_info = result.scalar()
        if enum_info and 'enum' in enum_info.lower():
            return [v.strip("'") for v in enum_info[enum_info.find('(')+1:enum_info.find(')')].split(',')]
    except Exception as e:
        print(f"  Could not get enum values: {e}")
    return None

def get_table_structure(conn, table_name):
    """Get detailed structure of a single table"""
    inspector = inspect(engine)
    
    # Get basic column info
    columns = []
    for column in inspector.get_columns(table_name):
        col_info = {
            'name': column['name'],
            'type': str(column['type']),
            'nullable': 'Yes' if column['nullable'] else 'No',
            'default': str(column.get('default', '')),
            'primary_key': 'Yes' if column.get('primary_key', False) else '',
        }
        
        # Get enum values if applicable
        if 'enum' in str(column['type']).lower():
            enum_vals = get_enum_values(conn, table_name, column['name'])
            if enum_vals:
                col_info['type'] = f"enum({', '.join(enum_vals)})"
        
        columns.append(col_info)
    
    # Get constraints
    constraints = []
    
    # Primary key
    pk = inspector.get_pk_constraint(table_name)
    if pk['constrained_columns']:
        constraints.append({
            'type': 'PRIMARY KEY',
            'columns': ', '.join(pk['constrained_columns']),
            'details': ''
        })
    
    # Foreign keys
    for fk in inspector.get_foreign_keys(table_name):
        constraints.append({
            'type': 'FOREIGN KEY',
            'columns': ', '.join(fk['constrained_columns']),
            'details': f"references {fk['referred_table']}({', '.join(fk['referred_columns'])})"
        })
    
    # Indexes
    for idx in inspector.get_indexes(table_name):
        constraints.append({
            'type': 'INDEX' + (' (UNIQUE)' if idx['unique'] else ''),
            'columns': ', '.join(idx['column_names']),
            'details': idx.get('name', '')
        })
    
    # Check constraints
    try:
        # First, get the table's OID
        oid_result = conn.execute(
            text("SELECT oid FROM pg_class WHERE relname = :table_name"),
            {"table_name": table_name}
        )
        table_oid = oid_result.scalar()
        
        if table_oid:
            # Now use the OID to get check constraints
            result = conn.execute(
                text("""
                    SELECT conname, pg_get_constraintdef(oid) 
                    FROM pg_constraint 
                    WHERE conrelid = :table_oid 
                    AND contype = 'c';
                """),
                {"table_oid": table_oid}
            )
            for name, definition in result:
                constraints.append({
                    'type': 'CHECK',
                    'columns': '',
                    'details': f"{name}: {definition}"
                })
    except Exception as e:
        print(f"  Could not get check constraints: {e}")
    
    return columns, constraints

def print_database_schema():
    """Print complete database schema"""
    inspector = inspect(engine)
    
    with engine.connect() as conn:
        # Get all tables
        tables = inspector.get_table_names()
        print(f"\n{' DATABASE SCHEMA ':=^80}\n")
        
        for table in tables:
            print(f"\n{' ' + table.upper() + ' ':-^80}")
            
            # Get table structure
            columns, constraints = get_table_structure(conn, table)
            
            # Print columns
            print("\nColumns:")
            print(tabulate(columns, headers='keys', tablefmt='grid'))
            
            # Print constraints if any
            if constraints:
                print("\nConstraints:")
                print(tabulate(constraints, headers='keys', tablefmt='grid'))
            
            # Get row count
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"\nRow count: {count:,}")
            except Exception as e:
                print(f"\nCould not get row count: {e}")
            
            print("\n" + "-"*80)

if __name__ == "__main__":
    try:
        print_database_schema()
    except Exception as e:
        print(f"Error: {e}")
        raise
