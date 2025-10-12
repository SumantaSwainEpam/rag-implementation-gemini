import sys
from rag_query import main as query_main
from ingest_docs import main as ingest_main

def main():
    while True:
        print("""
=== RAG with Gemini ===
1. Ingest documents
2. Ask a question
0. Exit
""")
        choice = input("Select option: ").strip()
        
        if choice == "1":
            print("\n🔄 Ingesting documents...")
            ingest_main()
            print("\n✅ Document ingestion completed!")
            input("Press Enter to continue...")
        elif choice == "2":
            print("\n🤔 Ask your question...")
            query_main()
            print("\n✅ Question answered!")
            input("Press Enter to continue...")
        elif choice == "0":
            print("\n👋 Bye!")
            break
        else:
            print("\n❌ Invalid option. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    main()
