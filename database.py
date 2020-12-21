import sqlite3


def push_data(a, b):
    M = sqlite3.connect('attend.db')
    cur = M.cursor()
    cur.execute("INSERT INTO name (Name, today) VALUES (?,?)", (a, b,))
    M.commit()
    M.close()
