import numpy as np
import librosa
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr

class ConfidenceDetector:
    def __init__(self):
        self.model = None
        self.feature_names = [
            'speech_rate', 'pitch_mean', 'pitch_std', 'pitch_range',
            'energy_mean', 'energy_std', 'filler_density', 'silence_ratio',
            'avg_silence_duration', 'voice_stability', 'articulation_rate'
        ]
        
    def extract_features(self, audio_path):
        """Extract confidence-related features from audio file"""
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # === Speech rate features ===
        # Get speech segments using VAD (Voice Activity Detection)
        speech_frames = librosa.effects.split(y, top_db=20)
        speech_duration = sum(end - start for start, end in speech_frames) / sr
        total_duration = len(y) / sr
        speech_rate = sum(1 for _ in speech_frames) / total_duration  # syllables/sec approximation
        
        # === Pitch features ===
        # Extract pitch (f0) using PYIN algorithm
        f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=400, sr=sr)
        f0_voiced = f0[voiced_flag]
        if len(f0_voiced) == 0:  # Handle silent recordings
            f0_voiced = np.array([0])
        
        pitch_mean = np.mean(f0_voiced)
        pitch_std = np.std(f0_voiced)
        pitch_range = np.max(f0_voiced) - np.min(f0_voiced) if len(f0_voiced) > 1 else 0
        
        # === Energy/volume features ===
        energy = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        # === Filler word detection === 
        # This would need a separate model in practice
        # Simplified approximation: detect short isolated voiced segments
        filler_count = 0
        min_filler_duration = 0.1  # seconds
        max_filler_duration = 0.5  # seconds
        
        for start, end in speech_frames:
            segment_duration = (end - start) / sr
            if min_filler_duration < segment_duration < max_filler_duration:
                segment = y[start:end]
                segment_energy = np.mean(librosa.feature.rms(y=segment)[0])
                # Low energy, short segments are potential fillers
                if segment_energy < 0.8 * energy_mean:
                    filler_count += 1
        
        filler_density = filler_count / total_duration
        
        # === Silence features ===
        silence_duration = total_duration - speech_duration
        silence_ratio = silence_duration / total_duration
        
        # Average silence duration
        silence_segments = []
        prev_end = 0
        for start, end in speech_frames:
            if start > prev_end:
                silence_segments.append((prev_end, start))
            prev_end = end
            
        if prev_end < len(y):
            silence_segments.append((prev_end, len(y)))
            
        avg_silence_duration = np.mean([end - start for start, end in silence_segments]) / sr if silence_segments else 0
        
        # === Voice stability ===
        # Jitter (cycle-to-cycle frequency variation)
        if len(f0_voiced) > 1:
            diffs = np.abs(np.diff(f0_voiced))
            voice_stability = 1.0 - (np.mean(diffs) / pitch_mean)
        else:
            voice_stability = 0
            
        # === Articulation rate ===
        # Speech rate during speaking segments only
        articulation_rate = speech_rate / (speech_duration / total_duration) if speech_duration > 0 else 0
        
        features = {
            'speech_rate': speech_rate,
            'pitch_mean': pitch_mean, 
            'pitch_std': pitch_std,
            'pitch_range': pitch_range,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'filler_density': filler_density,
            'silence_ratio': silence_ratio,
            'avg_silence_duration': avg_silence_duration,
            'voice_stability': voice_stability,
            'articulation_rate': articulation_rate
        }
        
        return features
    def train(self, audio_paths, confidence_scores):
        """Train the confidence detection model"""
        # Extract features from all audio files
        features_list = []
        for audio_path in audio_paths:
            features = self.extract_features(audio_path)
            features_list.append(features)
        
        # Create feature dataframe
        X = pd.DataFrame(features_list)
        y = np.array(confidence_scores)
        
        # Check if we have enough data for meaningful train/test split
        if len(audio_paths) < 3:
            print("Warning: Dataset too small for meaningful evaluation. Training on all data.")
            # Create and train model pipeline with all data
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            pipeline.fit(X, y)
            self.model = pipeline
            return None, None, None
        
        # If enough data, proceed with normal train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        y_pred = pipeline.predict(X_test)
        correlation, _ = pearsonr(y_test, y_pred)
        
        print(f"Model trained successfully")
        print(f"Train R² score: {train_score:.4f}")
        print(f"Test R² score: {test_score:.4f}")
        print(f"Correlation between predicted and actual scores: {correlation:.4f}")
        
        # Get feature importances
        importances = pipeline.named_steps['model'].feature_importances_
        feature_importance = {name: importance for name, importance in zip(self.feature_names, importances)}
        print("\nFeature importances:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        self.model = pipeline
        return train_score, test_score, correlation
    
    # def train(self, audio_paths, confidence_scores):
    #     """Train the confidence detection model"""
    #     # Extract features from all audio files
    #     features_list = []
    #     for audio_path in audio_paths:
    #         features = self.extract_features(audio_path)
    #         features_list.append(features)
        
    #     # Create feature dataframe
    #     X = pd.DataFrame(features_list)
    #     y = np.array(confidence_scores)
        
    #     # Split data
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    #     # Create and train model pipeline
    #     pipeline = Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    #     ])
        
    #     pipeline.fit(X_train, y_train)
        
    #     # Evaluate model
    #     train_score = pipeline.score(X_train, y_train)
    #     test_score = pipeline.score(X_test, y_test)
        
    #     y_pred = pipeline.predict(X_test)
    #     correlation, _ = pearsonr(y_test, y_pred)
        
    #     print(f"Model trained successfully")
    #     print(f"Train R² score: {train_score:.4f}")
    #     print(f"Test R² score: {test_score:.4f}")
    #     print(f"Correlation between predicted and actual scores: {correlation:.4f}")
        
    #     # Get feature importances
    #     importances = pipeline.named_steps['model'].feature_importances_
    #     feature_importance = {name: importance for name, importance in zip(self.feature_names, importances)}
    #     print("\nFeature importances:")
    #     for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
    #         print(f"{feature}: {importance:.4f}")
        
    #     self.model = pipeline
    #     return train_score, test_score, correlation
    
    def predict(self, audio_path):
        """Predict confidence score for a new audio recording"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Extract features
        features = self.extract_features(audio_path)
        features_df = pd.DataFrame([features])
        
        # Make prediction
        confidence_score = float(self.model.predict(features_df)[0])
        # Clip to 0-100 range and round to 2 decimal places
        confidence_score = round(max(0, min(100, confidence_score * 100)), 2)
        
        # Generate explanation
        explanation = self._generate_explanation(features)
        
        return {
            'confidence_score': confidence_score,
            'features': features,
            'explanation': explanation
        }
    
    def _generate_explanation(self, features):
        """Generate human-readable explanation for confidence assessment"""
        explanation = []
        
        # Speech rate assessment
        if features['speech_rate'] > 3.5:
            explanation.append("Speaking rate is very fast, which can indicate nervousness.")
        elif features['speech_rate'] < 1.5:
            explanation.append("Speaking rate is slow, which might indicate uncertainty.")
        else:
            explanation.append("Speaking rate is within a comfortable range.")
            
        # Filler words assessment
        if features['filler_density'] > 0.5:
            explanation.append("High frequency of filler words detected, suggesting hesitation.")
        elif features['filler_density'] < 0.2:
            explanation.append("Minimal use of filler words, indicating good preparation.")
            
        # Silence assessment
        if features['silence_ratio'] > 0.3:
            if features['avg_silence_duration'] > 1.0:
                explanation.append("Long pauses detected, potentially indicating thought collection or uncertainty.")
            else:
                explanation.append("Frequent short pauses detected, which may indicate thoughtful speaking.")
        else:
            explanation.append("Fluid speech with appropriate pausing.")
            
        # Pitch variation assessment
        if features['pitch_std'] > 50:
            explanation.append("High pitch variation indicates expressive and engaging speech.")
        elif features['pitch_std'] < 15:
            explanation.append("Low pitch variation suggests monotone delivery, which may indicate nervousness.")
            
        # Voice stability assessment
        if features['voice_stability'] > 0.9:
            explanation.append("Voice is very stable, indicating confidence.")
        elif features['voice_stability'] < 0.7:
            explanation.append("Voice shows some instability, which may indicate nervousness.")
            
        return explanation

# # Example usage
# if __name__ == "__main__":
#     # In a real-world scenario, you would have:
#     # 1. A labeled dataset of interview recordings with confidence ratings
#     # 2. A training pipeline
    
#     # Simulated dataset paths and labels (for demonstration)
#     audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]  # Replace with actual paths
#     confidence_scores = [0.85, 0.45]  # Expert-rated confidence scores (0-1)
    
#     # Create and train model
#     detector = ConfidenceDetector()
    
#     # In a real scenario, you would:
#     # detector.train(audio_paths, confidence_scores)
    
#     # For a new interview recording:
#     # result = detector.predict("path/to/new_interview.wav")
#     # print(f"Confidence Score: {result['confidence_score']}%")
#     # print("\nExplanation:")
#     # for point in result['explanation']:
#     #     print(f"- {point}")