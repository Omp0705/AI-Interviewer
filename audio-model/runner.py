from confidence_detector import ConfidenceDetector
import os

# Path to your sample audio file
sample_audio_path = "recorded_audio.wav"

# Create an instance of the detector
detector = ConfidenceDetector()

# Option 1: If you already have a trained model
# ----------------------------------------------
# If you've previously trained and saved the model:
# detector.model = load_your_saved_model()  # You'd need to implement model saving/loading

# Option 2: Train on sample data before prediction
# ------------------------------------------------
# For demonstration/testing, you can train on a tiny dataset:
# (In production, you'd train on a large, diverse dataset)

# Example of very simple training data (just for demonstration)
training_audio_files = ["HIGH_CNF.wav", "LOW_CNF.wav"]
confidence_labels = [0.9, 0.3]  # High confidence and low confidence examples

# Train the model
if all(os.path.exists(path) for path in training_audio_files):
    print("Training model on sample data...")
    detector.train(training_audio_files, confidence_labels)
else:
    print("Warning: Training files not found. Skipping training.")
    
# Run prediction on your sample file
if os.path.exists(sample_audio_path):
    print(f"\nAnalyzing sample audio: {sample_audio_path}")
    result = detector.predict(sample_audio_path)
    
    # Display results
    print(f"\nConfidence Score: {result['confidence_score']}%")
    print("\nExplanation:")
    for point in result['explanation']:
        print(f"- {point}")
        
    print("\nDetailed Features:")
    for feature, value in result['features'].items():
        print(f"- {feature}: {value:.4f}")
else:
    print(f"Error: Sample audio file not found at {sample_audio_path}")