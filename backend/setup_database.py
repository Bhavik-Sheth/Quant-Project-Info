"""
CF-AI-SDE Database Setup Script
================================

This script initializes the MongoDB database for CF-AI-SDE.
It creates collections, indexes, and verifies connectivity.

Run this after installing MongoDB and before using the system.

Usage:
    python setup_database.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    print("=" * 70)
    print("CF-AI-SDE Database Setup")
    print("=" * 70)
    
    # Load environment variables
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"\n✓ Loaded environment variables from {env_file}")
    else:
        print(f"\n⚠ No .env file found at {env_file}")
        print("  Using default MongoDB connection: mongodb://localhost:27017")
    
    # Get MongoDB connection
    print("\n1. Connecting to MongoDB...")
    try:
        from db.connection import get_mongodb_client, get_connection, setup_indexes
        
        connection = get_connection()
        
        if not connection.is_connected():
            print("❌ Failed to connect to MongoDB")
            print("\nPlease ensure:")
            print("  - MongoDB is running (check with: mongod --version)")
            print("  - MONGODB_URI is set correctly in .env file")
            print("  - Connection string format: mongodb://localhost:27017")
            print("\nIf using MongoDB Atlas:")
            print("  - Whitelist your IP address in Atlas dashboard")
            print("  - Use connection string format: mongodb+srv://...")
            return 1
        
        print("✓ Connected to MongoDB")
        
        # Show connection details
        db_name = os.getenv('MONGODB_DATABASE', 'cf_ai_sde')
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
        print(f"  Database: {db_name}")
        print(f"  URI: {mongodb_uri.split('@')[-1] if '@' in mongodb_uri else mongodb_uri}")
        
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\nTroubleshooting:")
        print("  1. Install pymongo: pip install pymongo")
        print("  2. Start MongoDB: mongod (or check your system service)")
        print("  3. Verify .env file contains correct MONGODB_URI")
        return 1
    
    # Setup indexes
    print("\n2. Creating collections and indexes...")
    try:
        client = get_mongodb_client()
        success = setup_indexes(client)
        
        if success:
            print("✓ Indexes created successfully")
        else:
            print("⚠ Some indexes may not have been created")
            
    except Exception as e:
        print(f"❌ Index creation error: {e}")
        return 1
    
    # Verify collections
    print("\n3. Verifying collections...")
    db_name = os.getenv('MONGODB_DATABASE', 'cf_ai_sde')
    
    client = get_mongodb_client()
    if client is None:
        print("❌ Cannot verify collections (database unavailable)")
        return 1
    
    expected_collections = [
        'market_data_raw',
        'market_data_validated',
        'market_data_clean',
        'market_features',
        'normalization_params',
        'validation_log',
        'agent_outputs',
        'agent_memory',
        'ml_models',
        'positions_and_risk'
    ]
    
    existing = client.list_collection_names()
    
    for collection in expected_collections:
        # Check if collection will be created (some are created on first write)
        doc_count = client[collection].estimated_document_count()
        if collection in existing or doc_count > 0:
            print(f"  ✓ {collection:<25} (documents: {doc_count})")
        else:
            print(f"  ℹ {collection:<25} (will be created on first write)")
    
    # Test write and read
    print("\n4. Testing database operations...")
    try:
        # Test write
        test_collection = client['_test']
        test_doc = {'test': 'data', 'timestamp': 'now'}
        test_collection.insert_one(test_doc)
        
        # Test read
        doc = test_collection.find_one({'test': 'data'})
        
        if doc and doc.get('test') == 'data':
            print("  ✓ Write operation successful")
            print("  ✓ Read operation successful")
        else:
            print("  ⚠ Read operation returned unexpected data")
        
        # Cleanup
        test_collection.delete_one({'test': 'data'})
        print("  ✓ Delete operation successful")
        
    except Exception as e:
        print(f"  ❌ Database operation error: {e}")
        return 1
    
    # Verify indexes
    print("\n5. Verifying indexes...")
    try:
        # Check indexes on a few key collections
        key_collections = ['market_data_raw', 'market_features', 'agent_outputs', 'ml_models']
        
        for coll_name in key_collections:
            indexes = list(client[coll_name].list_indexes())
            # Subtract 1 for default _id index
            index_count = len(indexes) - 1
            print(f"  ✓ {coll_name:<25} ({index_count} custom indexes)")
        
    except Exception as e:
        print(f"  ⚠ Could not verify all indexes: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ Database setup complete!")
    print("=" * 70)
    
    print("\n📊 Database Summary:")
    print(f"  • Database Name: {db_name}")
    print(f"  • Collections: {len(expected_collections)} configured")
    print(f"  • Status: Ready for use")
    
    print("\n🚀 Next Steps:")
    print("  1. Run health check:")
    print("     python -c \"from logical_pipe import TradingSystemAPI; api = TradingSystemAPI(); print(api.health_check())\"")
    print("\n  2. Ingest sample data:")
    print("     from logical_pipe import TradingSystemAPI")
    print("     api = TradingSystemAPI()")
    print("     data = api.run_partial_pipeline('ingest', {")
    print("         'symbols': ['AAPL'],")
    print("         'start_date': '2023-01-01',")
    print("         'end_date': '2023-12-31'")
    print("     })")
    print("\n  3. See README.md for full usage examples")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
