"""
Test script for improved audio enhancement
Tests: Noise reduction, Filler word removal, Timing preservation
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from services.video_service import AudioEnhancer
from pydub import AudioSegment
import time

def test_audio_enhancement():
    print("="*60)
    print("TESTING IMPROVED AUDIO ENHANCEMENT")
    print("="*60)
    
    # Find a test video file
    test_files = []
    uploads_dir = 'uploads'
    
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.endswith('.mp4'):
                test_files.append(os.path.join(uploads_dir, file))
    
    if not test_files:
        print("❌ No test video files found in uploads/")
        print("Please upload a video first!")
        return
    
    test_video = test_files[0]
    print(f"\n✅ Using test file: {test_video}")
    
    # Extract audio
    from moviepy.editor import VideoFileClip
    print("\n📹 Extracting audio from video...")
    video = VideoFileClip(test_video)
    audio_path = test_video.replace('.mp4', '_test_audio.wav')
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()
    
    print(f"✅ Audio extracted: {audio_path}")
    
    # Load original audio
    original_audio = AudioSegment.from_file(audio_path)
    original_duration = len(original_audio)
    print(f"\n📊 Original Audio:")
    print(f"   Duration: {original_duration}ms ({original_duration/1000:.2f}s)")
    print(f"   Sample Rate: {original_audio.frame_rate}Hz")
    print(f"   Channels: {original_audio.channels}")
    
    # Test different enhancement levels
    test_configs = [
        {
            'name': 'Conservative',
            'options': {
                'audio_enhancement_type': 'conservative',
                'pause_threshold': 500,
                'noise_reduction': 'light'
            }
        },
        {
            'name': 'Medium (Balanced)',
            'options': {
                'audio_enhancement_type': 'medium',
                'pause_threshold': 400,
                'noise_reduction': 'moderate'
            }
        },
        {
            'name': 'Aggressive',
            'options': {
                'audio_enhancement_type': 'aggressive',
                'pause_threshold': 300,
                'noise_reduction': 'strong'
            }
        }
    ]
    
    enhancer = AudioEnhancer()
    
    for config in test_configs:
        print("\n" + "="*60)
        print(f"TESTING: {config['name']} Enhancement")
        print("="*60)
        
        start_time = time.time()
        
        try:
            enhanced_audio, metrics = enhancer.enhance_audio(audio_path, config['options'])
            
            processing_time = time.time() - start_time
            
            print(f"\n✅ Enhancement Complete!")
            print(f"\n📊 Results:")
            print(f"   Processing Time: {processing_time:.2f}s")
            print(f"   Original Duration: {metrics['original_duration_ms']}ms ({metrics['original_duration_ms']/1000:.2f}s)")
            print(f"   Enhanced Duration: {metrics['enhanced_duration_ms']}ms ({metrics['enhanced_duration_ms']/1000:.2f}s)")
            print(f"   Time Saved: {metrics['time_saved_ms']}ms ({metrics['time_saved_percentage']:.1f}%)")
            print(f"   Filler Words Removed: {metrics['filler_words_removed']}")
            print(f"   Noise Reduction: {metrics['noise_reduction_level']}")
            
            # Calculate timing shift
            expected_reduction = metrics['time_saved_ms']
            actual_duration = len(enhanced_audio)
            timing_shift = abs(original_duration - expected_reduction - actual_duration)
            
            print(f"\n🎯 Timing Analysis:")
            print(f"   Expected Duration: {original_duration - expected_reduction}ms")
            print(f"   Actual Duration: {actual_duration}ms")
            print(f"   Timing Shift: {timing_shift}ms")
            
            if timing_shift < 100:
                print(f"   ✅ Excellent timing preservation!")
            elif timing_shift < 300:
                print(f"   ⚠️  Minor timing shift (acceptable)")
            else:
                print(f"   ❌ Significant timing shift detected")
            
            # Save enhanced audio for testing
            output_path = audio_path.replace('_test_audio.wav', f'_enhanced_{config["name"].lower().replace(" ", "_")}.wav')
            enhanced_audio.export(output_path, format='wav')
            print(f"\n💾 Saved: {output_path}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE!")
    print("="*60)
    print("\n📝 Summary:")
    print("   - Check timing preservation (should be < 300ms shift)")
    print("   - Listen to enhanced files for quality")
    print("   - Verify filler words are removed cleanly")
    print("   - Check background noise reduction effectiveness")
    
    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == '__main__':
    test_audio_enhancement()
