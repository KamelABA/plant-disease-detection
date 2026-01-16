"""
Data Migration Script: SQLite to MongoDB Atlas
This script migrates data from the existing SQLite database to MongoDB Atlas.

Usage:
    python migrate_to_mongodb.py
"""

import sqlite3
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# MongoDB Atlas Connection
# IMPORTANT: Set MONGODB_URI in your .env file
MONGODB_URI = os.getenv('MONGODB_URI')

if not MONGODB_URI:
    print("❌ MONGODB_URI environment variable is not set.")
    print("Please create a .env file with your MongoDB connection string.")
    print("See .env.example for the format.")
    exit(1)

def connect_sqlite():
    """Connect to SQLite database"""
    return sqlite3.connect('PlantDisease.db')

def connect_mongodb():
    """Connect to MongoDB Atlas"""
    client = MongoClient(MONGODB_URI)
    return client['PlantDisease']

def migrate_users(sqlite_conn, mongo_db):
    """Migrate users table to MongoDB"""
    print("Migrating users...")
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id, email, password, name FROM users")
    users = cursor.fetchall()
    
    if users:
        users_collection = mongo_db['users']
        for user in users:
            users_collection.update_one(
                {'email': user[1]},
                {'$set': {
                    'email': user[1],
                    'password': user[2],
                    'name': user[3] if len(user) > 3 else ''
                }},
                upsert=True
            )
        print(f"  Migrated {len(users)} users")
    else:
        print("  No users found")

def migrate_disease_info(sqlite_conn, mongo_db):
    """Migrate DiseaseInfromation table to MongoDB"""
    print("Migrating disease information...")
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id, disease_name, description, PossibleSteps FROM DiseaseInfromation")
    diseases = cursor.fetchall()
    
    if diseases:
        disease_collection = mongo_db['DiseaseInformation']
        for disease in diseases:
            possible_steps = disease[3]
            if isinstance(possible_steps, bytes):
                possible_steps = possible_steps.decode('utf-8', 'replace')
            
            disease_collection.update_one(
                {'disease_name': disease[1]},
                {'$set': {
                    'disease_name': disease[1],
                    'description': disease[2],
                    'PossibleSteps': possible_steps
                }},
                upsert=True
            )
        print(f"  Migrated {len(diseases)} disease records")
    else:
        print("  No disease info found")

def migrate_supplements(sqlite_conn, mongo_db):
    """Migrate Supplement table to MongoDB"""
    print("Migrating supplements...")
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT id, disease_name, supplementName, SupplementImage FROM Supplement")
    supplements = cursor.fetchall()
    
    if supplements:
        supplement_collection = mongo_db['Supplement']
        for supp in supplements:
            supplement_collection.update_one(
                {'disease_name': supp[1]},
                {'$set': {
                    'disease_name': supp[1],
                    'supplementName': supp[2],
                    'SupplementImage': supp[3] if len(supp) > 3 else ''
                }},
                upsert=True
            )
        print(f"  Migrated {len(supplements)} supplement records")
    else:
        print("  No supplements found")

def migrate_historique(sqlite_conn, mongo_db):
    """Migrate Historique table to MongoDB"""
    print("Migrating history...")
    cursor = sqlite_conn.cursor()
    try:
        cursor.execute("SELECT iduser, DiseaseName, Explanation, Recommendations, Supplement, date FROM Historique")
        history = cursor.fetchall()
        
        if history:
            history_collection = mongo_db['Historique']
            for record in history:
                history_collection.insert_one({
                    'iduser': str(record[0]),
                    'DiseaseName': record[1],
                    'Explanation': record[2],
                    'Recommendations': record[3],
                    'Supplement': record[4],
                    'date': record[5]
                })
            print(f"  Migrated {len(history)} history records")
        else:
            print("  No history found")
    except Exception as e:
        print(f"  Error migrating history: {e}")

def main():
    print("=" * 50)
    print("SQLite to MongoDB Atlas Migration")
    print("=" * 50)
    
    # Check if SQLite database exists
    if not os.path.exists('PlantDisease.db'):
        print("❌ SQLite database (PlantDisease.db) not found!")
        return
    
    # Connect to databases
    print("\nConnecting to SQLite...")
    sqlite_conn = connect_sqlite()
    
    print("Connecting to MongoDB Atlas...")
    try:
        mongo_db = connect_mongodb()
        # Test connection
        mongo_db.client.admin.command('ping')
        print("✅ Connected to MongoDB Atlas successfully!")
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB Atlas: {e}")
        return
    
    print("\nStarting migration...\n")
    
    # Migrate tables
    try:
        migrate_users(sqlite_conn, mongo_db)
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        migrate_disease_info(sqlite_conn, mongo_db)
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        migrate_supplements(sqlite_conn, mongo_db)
    except Exception as e:
        print(f"  Error: {e}")
    
    try:
        migrate_historique(sqlite_conn, mongo_db)
    except Exception as e:
        print(f"  Error: {e}")
    
    # Close connections
    sqlite_conn.close()
    
    print("\n" + "=" * 50)
    print("✅ Migration completed!")
    print("=" * 50)
    print("\nYou can now run the application with MongoDB Atlas:")
    print("  python app2.py")

if __name__ == "__main__":
    main()
