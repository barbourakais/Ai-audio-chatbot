#!/usr/bin/env python3
"""
Test script for vector database functionality
"""

import sys
import os
from vector_db import VectorDBManager
from utils import colored_print, setup_logging

logger = setup_logging()

def test_vector_database():
    """Test the vector database functionality"""
    try:
        colored_print("=== Testing Vector Database ===", "cyan", "bright")
        
        # Initialize vector database
        colored_print("Initializing vector database...", "cyan")
        vector_db = VectorDBManager()
        
        # Test database stats
        colored_print("\n1. Testing database statistics...", "cyan")
        stats = vector_db.get_database_stats()
        colored_print(f"Database stats: {stats}", "blue")
        
        # Test knowledge base update
        colored_print("\n2. Testing knowledge base update...", "cyan")
        if os.path.exists("Ox4labs.json"):
            num_chunks = vector_db.update_knowledge_base()
            colored_print(f"✓ Updated knowledge base with {num_chunks} chunks", "green")
        else:
            colored_print("⚠ Ox4labs.json not found, skipping update test", "yellow")
        
        # Test search functionality
        colored_print("\n3. Testing search functionality...", "cyan")
        test_queries = [
            "What services do you offer?",
            "Where are you located?",
            "How can I contact you?",
            "What is your process?",
            "AI consulting services"
        ]
        
        for query in test_queries:
            colored_print(f"\nSearching for: '{query}'", "blue")
            results = vector_db.search(query, top_k=2)
            
            if results:
                colored_print(f"Found {len(results)} results:", "green")
                for i, result in enumerate(results, 1):
                    colored_print(f"  {i}. Score: {result['similarity_score']:.3f}", "blue")
                    colored_print(f"     Text: {result['text'][:80]}...", "white")
            else:
                colored_print("No results found", "yellow")
        
        # Test context retrieval
        colored_print("\n4. Testing context retrieval...", "cyan")
        context = vector_db.get_context_for_query("AI services", top_k=3)
        if context:
            colored_print(f"Context length: {len(context)} characters", "green")
            colored_print(f"Context preview: {context[:100]}...", "white")
        else:
            colored_print("No context found", "yellow")
        
        colored_print("\n✓ All vector database tests completed successfully!", "green", "bright")
        return True
        
    except Exception as e:
        colored_print(f"✗ Error during vector database testing: {e}", "red")
        logger.error(f"Vector database test error: {e}")
        return False

def test_llm_integration():
    """Test LLM integration with vector database"""
    try:
        colored_print("\n=== Testing LLM Integration ===", "cyan", "bright")
        
        from llm_client import OllamaClient
        
        # Initialize LLM client
        colored_print("Initializing LLM client...", "cyan")
        llm = OllamaClient()
        
        # Test knowledge base stats
        colored_print("\n1. Testing LLM knowledge base stats...", "cyan")
        stats = llm.get_knowledge_base_stats()
        colored_print(f"LLM knowledge base stats: {stats}", "blue")
        
        # Test search through LLM
        colored_print("\n2. Testing LLM search...", "cyan")
        results = llm.search_knowledge_base("AI services", top_k=2)
        if results:
            colored_print(f"Found {len(results)} results through LLM", "green")
        else:
            colored_print("No results found through LLM", "yellow")
        
        # Test response generation with context
        colored_print("\n3. Testing response generation with context...", "cyan")
        response = llm.generate_response("What services do you offer?")
        if response:
            colored_print(f"Generated response: {response}", "green")
        else:
            colored_print("No response generated", "yellow")
        
        colored_print("\n✓ All LLM integration tests completed successfully!", "green", "bright")
        return True
        
    except Exception as e:
        colored_print(f"✗ Error during LLM integration testing: {e}", "red")
        logger.error(f"LLM integration test error: {e}")
        return False

def main():
    """Main test function"""
    try:
        colored_print("Starting vector database tests...", "cyan", "bright")
        
        # Test vector database
        vector_success = test_vector_database()
        
        # Test LLM integration
        llm_success = test_llm_integration()
        
        # Summary
        colored_print("\n=== Test Summary ===", "cyan", "bright")
        colored_print(f"Vector Database: {'✓ PASS' if vector_success else '✗ FAIL'}", 
                     "green" if vector_success else "red")
        colored_print(f"LLM Integration: {'✓ PASS' if llm_success else '✗ FAIL'}", 
                     "green" if llm_success else "red")
        
        if vector_success and llm_success:
            colored_print("\n🎉 All tests passed! Vector database is working correctly.", "green", "bright")
            return 0
        else:
            colored_print("\n❌ Some tests failed. Please check the errors above.", "red")
            return 1
            
    except Exception as e:
        colored_print(f"✗ Fatal error during testing: {e}", "red")
        logger.error(f"Fatal test error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 