"""
Master Entry Point for the Lumbar Spinal Stenosis (LSS) AI Diagnostic & Benchmark Platform.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.config import CONFIG
from data.synthetic_data_generator import generate_mock_rsna_dataset
from training.benchmark_runner import run_lss_benchmark
from inference.predict import LSSClinicalPredictor

def main():
    parser = argparse.ArgumentParser(description="Lumbar Spinal Stenosis (LSS) AI Platform")
    parser.add_argument('--generate-data', action='store_true', help="Generate synthetic RSNA-compliant DICOM dataset")
    parser.add_argument('--num-patients', type=int, default=30, help="Number of synthetic patients to generate")
    parser.add_argument('--benchmark', action='store_true', help="Run full benchmark of 5 Image & 5 Tabular models")
    parser.add_argument('--epochs', type=int, default=4, help="Number of training epochs per benchmark model")
    parser.add_argument('--predict', type=str, default=None, help="Path to patient DICOM or image file for inference")
    parser.add_argument('--web', action='store_true', help="Launch interactive web application")
    parser.add_argument('--port', type=int, default=5000, help="Port for web server")

    args = parser.parse_args()

    # Step 1: Data Check or Generation
    if args.generate_data or not (CONFIG.train_csv_path.exists() and any(CONFIG.raw_dicom_dir.glob("**/*.dcm"))):
        print("\n[Data Preparation] Generating realistic RSNA Lumbar Spine DICOM dataset...")
        generate_mock_rsna_dataset(num_patients=args.num_patients)

    # Step 2: Prediction Mode
    if args.predict:
        pred_path = Path(args.predict)
        if not pred_path.exists():
            print(f"Error: File {pred_path} does not exist.")
            sys.exit(1)
        predictor = LSSClinicalPredictor()
        finding = predictor.predict_slice(pred_path)
        print("\n" + "=" * 50)
        print("  CLINICAL INFERENCE RESULT")
        print("=" * 50)
        print(f"Target Level:        {finding['level']}")
        print(f"Predicted Severity:  {finding['predicted_severity']}")
        print(f"Confidence:          {finding['confidence'] * 100:.2f}%")
        print("Class Probabilities:")
        for cls_name, prob in finding['probabilities'].items():
            print(f"  - {cls_name:12s}: {prob * 100:.2f}%")
        print("=" * 50)
        return

    # Step 3: Web Server Mode
    if args.web:
        from web.app import app
        print(f"\nLaunching LSS Web Dashboard on http://localhost:{args.port}...")
        app.run(host='0.0.0.0', port=args.port, debug=False)
        return

    # Step 4: Benchmark Mode (default execution if neither predict nor web is specified)
    print("\nExecuting Comprehensive Model Benchmark & Top 3 Comparison...")
    run_lss_benchmark(num_epochs=args.epochs)

if __name__ == "__main__":
    main()
