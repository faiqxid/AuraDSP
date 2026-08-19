import sys
from pathlib import Path
import os
import shutil

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute() / "src"))

from detector import AIMusicDetector
from remaster import RemasterPipeline

def run_test():
    base_dir = Path(__file__).parent.parent.absolute()
    input_file = str(base_dir / "aigent.mp3")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)
        
    print(f"1. Testing AI Music Detector on: {input_file}")
    detector = AIMusicDetector()
    res_before = detector.detect(input_file)
    print(f"  AI Probability Before: {res_before.percentage}%")
    print(f"  Verdict: {res_before.verdict}")

    print("\n2. Testing Remaster Pipeline (Aggressive Mode)...")
    output_dir = base_dir / "tests_output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()
    
    pipeline = RemasterPipeline(output_dir=output_dir, intensity="aggressive")
    remaster_res = pipeline.remaster(input_file, output_name="remastered_aigent.wav")
    print(f"  Remaster finished. Output file: {remaster_res.filepath}")
    
    print("\n3. Testing AI Music Detector on Remastered Output...")
    res_after = detector.detect(remaster_res.filepath)
    print(f"  AI Probability After: {res_after.percentage}%")
    print(f"  Verdict: {res_after.verdict}")
    
    # Assert bypass succeeds (AI probability < 10%)
    success = res_after.percentage < 10.0
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
    
    # Cleanup tests_output
    if success:
        try:
            shutil.rmtree(output_dir)
            print("Cleaned up test output folder.")
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")
            
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    run_test()
