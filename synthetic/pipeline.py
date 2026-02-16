import subprocess
import sys
import time

def run_step(script_name):
    print(f"\n[Pipeline] Running {script_name}...")
    try:
        # Run python script in logical order
        subprocess.check_call([sys.executable, script_name])
        print(f"[Pipeline] {script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[Pipeline] Error running {script_name}: {e}")
        sys.exit(1)

def main():
    print("=========================================")
    print("   MindMate Synthetic Data Pipeline      ")
    print("=========================================")
    
    start_time = time.time()
    
    # 1. Generate Scenarios
    run_step("generate_scenarios.py")
    
    # 2. Process Dialogues (Generate + Filter + Critique)
    run_step("process_dialogues.py")
    
    elapsed = time.time() - start_time
    print(f"\n[Pipeline] Pipeline finished in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
