"""
Evaluation logic largely depends on the spider evaluation script (exec_eval).
"""
import os
import json
import re
import sqlite3
import itertools
from typing import List, Dict, Tuple, Any, Set
from collections import defaultdict
import random

class sqlTester:
    """
    SQL execution tester for comparing and evaluating SQL queries.
    Provides database schema overview and SQL execution comparison capabilities.
    """
    
    def __init__(self, 
                 tables_file: str = "tables.json",
                 databases_dir: str = "databases"):
        """
        Initialize SQL tester with database schemas and execution logic
        
        Args:
            tables_file: Path to JSON file containing database schemas
            databases_dir: Directory containing SQLite database files
        """
        self.tables_file = tables_file
        self.databases_dir = databases_dir
        
        # Load database schemas from tables file
        if os.path.exists(tables_file):
            with open(tables_file, 'r', encoding='utf-8') as f:
                self.tables = {db['db_id']: db for db in json.load(f)}
        else:
            self.tables = {}
    
    def extract_dbid(self, question: str) -> str:
        """Extract database ID from question.
        
        Args:
            question: String in format "dbid|question"
            
        Returns:
            Database ID string
        """
        if "|" in question:
            return question.split("|")[0]
        return question

    def get_database_overview(self, db_id: str) -> str:
        """
        Generate formatted overview of database structure
        
        Args:
            db_id: Database identifier
            
        Returns:
            Formatted string containing database schema information
        """
        if db_id not in self.tables:
            return f"Database {db_id} not found in tables data"
        
        db_info = self.tables[db_id]
        overview_parts = []
        
        # Database overview section
        db_overview = db_info.get('db_overview', 'No overview available')
        overview_parts.append(f"Database Overview: {db_overview}")
        overview_parts.append("")
        
        # Table listing
        overview_parts.append("Tables:")
        table_names = db_info.get('table_names_original', [])
        for table_name in table_names:
            overview_parts.append(f"- {table_name}")
        overview_parts.append("")
        
        # Column details
        overview_parts.append("Columns:")
        column_names_original = db_info.get('column_names_original', [])
        column_descriptions = db_info.get('column_descriptions', [])
        column_types = db_info.get('column_types', [])
        
        # Process each column
        for i, column_info in enumerate(column_names_original):
            if i == 0:  # Skip index column
                continue
                
            table_idx, column_name = column_info
            if table_idx < 0 or table_idx >= len(table_names):
                continue
                
            table_name = table_names[table_idx]
            
            # Build column description
            description = ""
            if i < len(column_descriptions) and column_descriptions[i]:
                description = column_descriptions[i]
            elif i < len(column_types) and column_types[i]:
                description = f"{column_types[i]} type"
            else:
                description = "No description available"
            
            overview_parts.append(f"- {table_name}.{column_name}: {description}")
        
        return "\n".join(overview_parts)

    # ==================== Query Result Comparison ====================
    
    def permute_tuple(self, element: Tuple, perm: Tuple) -> Tuple:
        """Reorder tuple elements based on permutation indices"""
        assert len(element) == len(perm)
        return tuple([element[i] for i in perm])

    def unorder_row(self, row: Tuple) -> Tuple:
        """Create sorted version of row for comparison"""
        return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))

    def quick_rej(self, result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
        """Quick comparison using unordered rows"""
        s1 = [self.unorder_row(row) for row in result1]
        s2 = [self.unorder_row(row) for row in result2]
        if order_matters:
            return s1 == s2
        else:
            return set(s1) == set(s2)

    def multiset_eq(self, l1: List, l2: List) -> bool:
        """Check if two lists contain same elements with same frequencies"""
        if len(l1) != len(l2):
            return False
        d = defaultdict(int)
        for e in l1:
            d[e] = d[e] + 1
        for e in l2:
            d[e] = d[e] - 1
            if d[e] < 0:
                return False
        return True

    def get_constraint_permutation(self, tab1_sets_by_columns: List[Set], result2: List[Tuple]):
        """Generate column permutations considering value constraints"""
        num_cols = len(result2[0])
        perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
        if num_cols <= 3:
            return itertools.product(*perm_constraints)

        # Use random sampling to reduce permutation space
        for _ in range(min(20, len(result2))):
            random_tab2_row = random.choice(result2)
            for tab1_col in range(num_cols):
                for tab2_col in set(perm_constraints[tab1_col]):
                    if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                        perm_constraints[tab1_col].remove(tab2_col)
        return itertools.product(*perm_constraints)

    def result_eq(self, result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
        """Compare query results with column permutation support"""
        if len(result1) == 0 and len(result2) == 0:
            return True

        if len(result1) != len(result2):
            return False

        num_cols = len(result1[0])
        if len(result2[0]) != num_cols:
            return False

        # Initial quick comparison
        if not self.quick_rej(result1, result2, order_matters):
            return False
        
        if result1 == result2:
            return True
            
        # Skip large result sets for performance
        if len(result2) > 200 or len(result1) > 200:
            return False

        tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

        # Test different column orderings
        for perm in self.get_constraint_permutation(tab1_sets_by_columns, result2):
            if len(perm) != len(set(perm)):
                continue
            if num_cols == 1:
                result2_perm = result2
            else:
                result2_perm = [self.permute_tuple(element, perm) for element in result2]
            
            if order_matters:
                if result1 == result2_perm:
                    return True
            else:
                if set(result1) == set(result2_perm) and self.multiset_eq(result1, result2_perm):
                    return True
        return False

    # ==================== SQL Query Processing ====================

    def postprocess_query(self, query: str) -> str:
        """Fix common SQL formatting issues"""
        query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
        return query

    def remove_distinct(self, s: str) -> str:
        """Remove DISTINCT keyword from SQL query"""
        return re.sub(r'\bDISTINCT\b', '', s, flags=re.IGNORECASE)

    def replace_cur_year(self, query: str) -> str:
        """Replace dynamic year functions with static value"""
        return re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE)

    def get_cursor_from_path(self, sqlite_path: str):
        """Create database connection cursor"""
        try:
            if not os.path.exists(sqlite_path):
                print(f"Opening a new connection: {sqlite_path}")
            connection = sqlite3.connect(sqlite_path)
            connection.text_factory = lambda b: b.decode(errors="ignore")
            return connection.cursor()
        except Exception as e:
            print(f"Error connecting to {sqlite_path}: {e}")
            raise e

    def exec_on_db_sync(self, sqlite_path: str, query: str) -> Tuple[str, Any]:
        """Execute SQL query and return results"""
        query = self.replace_cur_year(query)
        cursor = self.get_cursor_from_path(sqlite_path)
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            cursor.connection.close()
            return "result", result
        except Exception as e:
            cursor.connection.close()
            return "exception", e

    # ==================== SQL Extraction from Text ====================

    def extract_sql_from_answer(self, answer: str) -> str:
        """Extract SQL query from text response"""
        answer = answer.strip()

        # Check for SQL code blocks
        code_block = re.search(r"```sql(.*?)```", answer, re.DOTALL | re.IGNORECASE)
        if code_block:
            sql = code_block.group(1).strip()
            return self._extract_complete_sql(sql)

        # Find SQL keywords in text
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "WITH"]
        positions = [(kw, answer.upper().find(kw)) for kw in sql_keywords]
        positions = [(kw, pos) for kw, pos in positions if pos != -1]

        if not positions:
            return answer

        _, start = min(positions, key=lambda x: x[1])
        sql_text = answer[start:].strip()

        return self._extract_complete_sql(sql_text)

    def _extract_complete_sql(self, sql: str) -> str:
        """Extract complete SQL statement handling quotes and semicolons"""
        result = []
        in_single_quote = False
        in_double_quote = False

        for ch in sql:
            result.append(ch)

            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif ch == ";" and not in_single_quote and not in_double_quote:
                break

        cleaned = "".join(result).strip()

        # Close any open quotes
        if in_single_quote:
            cleaned += "'"
        if in_double_quote:
            cleaned += '"'

        # Remove trailing semicolon
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()

        return cleaned

    # ==================== Database Management ====================

    def get_database_paths(self, db_id: str) -> List[str]:
        """Find all database files for given database ID"""
        # Check primary database path
        db_path = os.path.join(self.databases_dir, f"{db_id}/{db_id}.sqlite")
        if os.path.exists(db_path):
            db_dir = os.path.dirname(db_path)
        else:
            # Check alternative path
            db_path = os.path.join(self.databases_dir, f"{db_id}.sqlite")
            if os.path.exists(db_path):
                db_dir = os.path.dirname(db_path)
            else:
                return []

        # Find all SQLite files in directory
        db_paths = []
        for item in os.listdir(db_dir):
            if item.endswith('.sqlite'):
                db_paths.append(os.path.join(db_dir, item))
        
        return db_paths if db_paths else [db_path]

    # ==================== Main Evaluation Method ====================

    def evaluate_execution(self, db_id: str, predicted_sql: str, gold_sql: str) -> bool:
        """
        Compare predicted SQL against gold standard by execution results
        
        Args:
            db_id: Target database identifier
            predicted_sql: Generated SQL query to test
            gold_sql: Reference SQL query
            
        Returns:
            True if queries produce equivalent results, False otherwise
        """
        # Normalize query formatting
        p_str = self.postprocess_query(predicted_sql)
        g_str = self.postprocess_query(gold_sql)
        
        # Remove DISTINCT keyword
        p_str = self.remove_distinct(p_str)
        g_str = self.remove_distinct(g_str)

        # Determine if row order affects correctness
        order_matters = 'order by' in g_str.lower()

        # Locate database files
        db_paths = self.get_database_paths(db_id)
        if not db_paths:
            print(f"No databases found for {db_id}")
            return False

        # Execute and compare on all database instances
        for db_path in db_paths:
            # Execute reference query
            g_flag, g_denotation = self.exec_on_db_sync(db_path, g_str)
            if g_flag == 'exception':
                print(f"Reference query failed on {db_path}: {g_denotation}")
                return False

            # Execute predicted query
            p_flag, p_denotation = self.exec_on_db_sync(db_path, p_str)
            if p_flag == 'exception':
                return False

            # Compare execution results
            if not self.result_eq(g_denotation, p_denotation, order_matters):
                return False

        return True