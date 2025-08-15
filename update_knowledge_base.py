#!/usr/bin/env python3
"""
Simple script to update the knowledge base when JSON content changes
"""

from vector_db import VectorDBManager
from utils import colored_print, setup_logging

logger = setup_logging()

def update_knowledge_base():
    """Update the knowledge base with latest JSON content"""
    try:
        colored_print("=== Updating Knowledge Base ===", "cyan", "bright")
        
        # Initialize vector database
        vector_db = VectorDBManager()
        
        # Update the knowledge base
        success = vector_db.update_knowledge_base("Ox4labs.json")
        
        if success:
            # Get updated stats
            stats = vector_db.get_database_stats()
            colored_print(f"\n✓ Knowledge base updated successfully!", "green", "bright")
            colored_print(f"Total chunks: {stats.get('total_chunks', 0)}", "green")
            colored_print(f"Collection: {stats.get('collection_name', 'unknown')}", "green")
            return True
        else:
            colored_print("✗ Failed to update knowledge base", "red")
            return False
            
    except Exception as e:
        colored_print(f"✗ Error updating knowledge base: {e}", "red")
        return False

def main():
    """Main function"""
    try:
        success = update_knowledge_base()
        
        if success:
            colored_print("\n🎉 Knowledge base update completed!", "green", "bright")
            colored_print("The AI assistant will now use the updated content.", "green")
            return 0
        else:
            colored_print("\n❌ Knowledge base update failed", "red")
            return 1
            
    except Exception as e:
        colored_print(f"Fatal error: {e}", "red")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 