import sqlite3

class MemoryManager:
    def __init__(self, db_path="data/user_data.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            user_id TEXT,
            context TEXT
        )""")

    def save_context(self, user_id, context):
        self.conn.execute("DELETE FROM memory WHERE user_id = ?", (user_id,))
        self.conn.execute("INSERT INTO memory (user_id, context) VALUES (?, ?)", (user_id, context))
        self.conn.commit()

    def get_context(self, user_id):
        cursor = self.conn.execute("SELECT context FROM memory WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return row[0] if row else ""
