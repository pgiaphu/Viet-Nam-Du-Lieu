import sys
import os

# Create a robust path to the src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from sql_agent_system.crew import SqlAgentSystemCrew

def run():
    """
    Run the crew using interactive input.
    """
    print("\n" + "="*60)
    print("🚀 PRODUCTION SQL AGENT SYSTEM")
    print("="*60)
    print("✅ Features:")
    print("   • Standard CrewAI Structure")
    print("   • Modular Tools & Configuration")
    print("   • Secure Environment Variables")
    print("-" * 60)
    
    try:
        while True:
            # Check for command line args for non-interactive mode
            if len(sys.argv) > 1:
                query = " ".join(sys.argv[1:])
                print(f"Processing command line argument: {query}")
                result = SqlAgentSystemCrew().crew().kickoff(inputs={'query': query})
                print(str(result))
                return

            user_query = input("\n❓ Your Question (or 'exit'): ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
                
            if not user_query:
                continue
                
            print(f"\n🧠 Processing: {user_query}...")
            try:
                result = SqlAgentSystemCrew().crew().kickoff(inputs={'query': user_query})
                print("\n" + "="*60)
                print(f"🎯 FINAL RESULT:\n{result}")
                print("="*60)
            except KeyboardInterrupt:
                print("\n\n⚠️  Process interrupted by user")
                break
            except Exception as inner_e:
                print(f"\n❌ Crew Execution Error: {inner_e}")
                import traceback
                traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run()
