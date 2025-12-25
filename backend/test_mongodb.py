from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get MongoDB URI
mongodb_uri = os.getenv('MONGODB_ATLAS_URI')

print("Testing MongoDB Atlas connection...")
print(f"URI: {mongodb_uri[:50]}...") # Show first 50 chars only

try:
    # Try to connect
    client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
    
    # Test the connection
    client.admin.command('ping')
    
    print("✅ SUCCESS! Connected to MongoDB Atlas!")
    
    # Show databases
    print("\nAvailable databases:")
    for db_name in client.list_database_names():
        print(f"  - {db_name}")
    
    # Check if snipx database exists
    db = client['snipx']
    print(f"\nCollections in 'snipx' database:")
    for collection in db.list_collection_names():
        print(f"  - {collection}")
    
    if not db.list_collection_names():
        print("  (No collections yet - will be created when you add data)")
    
    client.close()
    
except Exception as e:
    print(f"❌ ERROR: Could not connect to MongoDB Atlas")
    print(f"Error details: {str(e)}")
    print("\nCommon fixes:")
    print("1. Check username and password in connection string")
    print("2. Make sure IP address is whitelisted (0.0.0.0/0)")
    print("3. Check if cluster is still being created (wait 5 minutes)")
    print("4. Verify connection string format")
