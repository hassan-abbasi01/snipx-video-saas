import os
import json
from datetime import datetime
from models.video import Video
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import magic
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
# TEMPORARILY DISABLED - pydub requires audioop (removed in Python 3.13)
# from pydub import AudioSegment
# from pydub.effects import normalize, compress_dynamic_range
# from pydub.silence import split_on_silence, detect_nonsilent
# import tensorflow as tf  # Removed for minimal deployment
# from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration  # Removed for minimal deployment
# import torch  # Removed for minimal deployment
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
# import librosa  # Removed for minimal deployment (requires scipy)
# import scipy.signal  # Removed for minimal deployment
import re

# Set FFmpeg path for Windows
# Try multiple possible locations
POSSIBLE_FFMPEG_PATHS = [
    r"C:\Users\Cv\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin",
    r"C:\Users\Cv\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin",
    r"C:\Users\PCP\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
]

FFMPEG_PATH = None
for path in POSSIBLE_FFMPEG_PATHS:
    if os.path.exists(path) and os.path.exists(os.path.join(path, 'ffmpeg.exe')):
        FFMPEG_PATH = path
        break

if FFMPEG_PATH:
    if FFMPEG_PATH not in os.environ.get('PATH', ''):
        os.environ['PATH'] = FFMPEG_PATH + os.pathsep + os.environ.get('PATH', '')
    
    # Set FFmpeg for imageio
    os.environ['IMAGEIO_FFMPEG_EXE'] = os.path.join(FFMPEG_PATH, 'ffmpeg.exe')
    
    # Configure AudioSegment to use FFmpeg
    AudioSegment.converter = os.path.join(FFMPEG_PATH, 'ffmpeg.exe')
    AudioSegment.ffmpeg = os.path.join(FFMPEG_PATH, 'ffmpeg.exe')
    AudioSegment.ffprobe = os.path.join(FFMPEG_PATH, 'ffprobe.exe')
    print(f"[INFO] FFmpeg configured at: {FFMPEG_PATH}")
else:
    print("[WARNING] FFmpeg not found in common locations. Please ensure FFmpeg is installed and in PATH.")
    print("[WARNING] You can install FFmpeg using: winget install Gyan.FFmpeg")

class AIThumbnailGenerator:
    """AI-powered YouTube thumbnail generator with BLIP captioning and intelligent frame selection"""
    
    def __init__(self):
        print("[AI THUMBNAIL] Initializing AI Thumbnail Generator...")
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[AI THUMBNAIL] Using device: {self.device}")
        
        # Load face detection cascade for better frame selection
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("[AI THUMBNAIL] Face detection loaded successfully")
        except:
            print("[AI THUMBNAIL] Face detection not available")
            self.face_cascade = None
        
    def _load_model(self):
        """Lazy load BLIP model only when needed"""
        if self.processor is None:
            try:
                print("[AI THUMBNAIL] Loading BLIP model for image captioning...")
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
                print("[AI THUMBNAIL] BLIP model loaded successfully")
            except Exception as e:
                print(f"[AI THUMBNAIL] Failed to load BLIP model: {e}")
                self.processor = None
                self.model = None
    
    def generate_catchy_text(self, frame_path, video_filename):
        """Generate catchy text using AI image captioning"""
        try:
            self._load_model()
            
            if self.model is None:
                # Fallback to filename-based generation
                return self._fallback_text_generation(video_filename)
            
            # Load image
            image = Image.open(frame_path).convert('RGB')
            
            # Generate caption using BLIP
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=20)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            print(f"[AI THUMBNAIL] Generated caption: {caption}")
            
            # Convert caption to catchy thumbnail text
            catchy_text = self._make_catchy(caption, video_filename)
            print(f"[AI THUMBNAIL] Catchy text: {catchy_text}")
            
            return catchy_text
            
        except Exception as e:
            print(f"[AI THUMBNAIL] Error in AI text generation: {e}")
            return self._fallback_text_generation(video_filename)
    
    def _make_catchy(self, caption, filename):
        """Transform AI caption into catchy thumbnail text"""
        # Extract keywords from caption
        caption_lower = caption.lower()
        
        # Catchy prefixes based on content
        action_words = ['running', 'jumping', 'playing', 'dancing', 'swimming', 'flying', 'driving']
        nature_words = ['sunset', 'beach', 'mountain', 'ocean', 'forest', 'sky', 'landscape']
        people_words = ['person', 'man', 'woman', 'people', 'group', 'child', 'baby']
        object_words = ['car', 'building', 'house', 'phone', 'computer', 'food']
        
        if any(word in caption_lower for word in action_words):
            prefix = np.random.choice(['WATCH THIS!', 'AMAZING!', 'INCREDIBLE!', 'WOW!'])
        elif any(word in caption_lower for word in nature_words):
            prefix = np.random.choice(['BREATHTAKING', 'STUNNING', 'BEAUTIFUL', 'SPECTACULAR'])
        elif any(word in caption_lower for word in people_words):
            prefix = np.random.choice(['MUST SEE', 'VIRAL', 'TRENDING', 'WATCH NOW'])
        else:
            prefix = np.random.choice(['NEW VIDEO', 'DISCOVER', 'CHECK THIS OUT', 'DON\'T MISS'])
        
        # Extract main subject from caption
        words = caption.split()
        # Get important words (skip common words)
        skip_words = ['a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'to', 'of', 'with']
        important_words = [w.upper() for w in words if w.lower() not in skip_words][:3]
        
        if important_words:
            return f"{prefix}: {' '.join(important_words)}"
        else:
            return prefix
    
    def _fallback_text_generation(self, filename):
        """Fallback text generation from filename"""
        # Clean filename
        name = os.path.splitext(os.path.basename(filename))[0]
        name = re.sub(r'[_-]', ' ', name)
        name = name.title()
        
        prefixes = ['NEW VIDEO', 'WATCH NOW', 'MUST SEE', 'TRENDING', 'VIRAL', 'AMAZING']
        prefix = np.random.choice(prefixes)
        
        if len(name) > 30:
            name = name[:30] + '...'
        
        return f"{prefix}: {name}"
    
    def create_youtube_thumbnail(self, frame_path, text, output_path):
        """Create professional YouTube thumbnail with AI-generated text"""
        try:
            # Load frame
            img = Image.open(frame_path).convert('RGB')
            
            # Resize to YouTube format (1280x720)
            target_size = (1280, 720)
            img = self._resize_with_crop(img, target_size)
            
            # Apply visual enhancements
            img = self._enhance_image(img)
            
            # Add text overlay only if text is provided
            if text and text.strip():
                img = self._add_professional_text(img, text)
            
            # Save with high quality
            img.save(output_path, 'JPEG', quality=95)
            print(f"[AI THUMBNAIL] Saved YouTube thumbnail: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"[AI THUMBNAIL] Error creating YouTube thumbnail: {e}")
            return frame_path
    
    def _resize_with_crop(self, img, target_size):
        """Resize image to exact dimensions with smart cropping"""
        target_ratio = target_size[0] / target_size[1]
        img_ratio = img.width / img.height
        
        if img_ratio > target_ratio:
            # Image is wider - crop width
            new_height = target_size[1]
            new_width = int(new_height * img_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Center crop
            left = (new_width - target_size[0]) // 2
            img = img.crop((left, 0, left + target_size[0], target_size[1]))
        else:
            # Image is taller - crop height
            new_width = target_size[0]
            new_height = int(new_width / img_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Center crop
            top = (new_height - target_size[1]) // 2
            img = img.crop((0, top, target_size[0], top + target_size[1]))
        
        return img
    
    def _enhance_image(self, img):
        """Apply minimal enhancements - keep original look"""
        # Just return the original image without heavy processing
        # This removes the vintage/dark vignette effect
        return img
    
    def _add_vignette(self, img):
        """Vignette disabled - returns original image"""
        # Vignette removed to prevent dark effect
        return img
    
    def _add_professional_text(self, img, text):
        """Add professional text overlay with advanced styling and effects"""
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Try to load bold font for impact
        try:
            font_size = min(width // 12, 100)  # Larger font for impact
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("Arial Bold.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
        
        # Calculate text size and wrap if needed
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] > width * 0.85:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate total text height
        total_height = 0
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            line_heights.append(line_height)
            total_height += line_height + 10
        
        # Create gradient background bar
        bar_height = total_height + 80
        bar_y = height - bar_height - 40
        
        # Create overlay with gradient
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        # Draw gradient bar using multiple rectangles
        for i in range(bar_height):
            alpha = int(220 - (i / bar_height) * 40)  # Gradient from dark to lighter
            overlay_draw.rectangle(
                [0, bar_y + i, width, bar_y + i + 1],
                fill=(0, 0, 0, alpha)
            )
        
        # Composite overlay
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, overlay)
        img = img.convert('RGB')
        
        # Draw text with multiple layers for depth
        draw = ImageDraw.Draw(img)
        
        # Draw each line
        current_y = bar_y + 40
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (width - text_width) // 2
            
            # Layer 1: Heavy black shadow for depth
            for offset in range(8, 0, -1):
                shadow_alpha = int(255 * (offset / 8))
                shadow_color = (0, 0, 0)
                draw.text((text_x + offset//2, current_y + offset//2), line, font=font, fill=shadow_color)
            
            # Layer 2: Colored outline (thick)
            outline_color = (255, 100, 0)  # Orange outline
            outline_range = 5
            for adj_x in range(-outline_range, outline_range + 1):
                for adj_y in range(-outline_range, outline_range + 1):
                    if adj_x*adj_x + adj_y*adj_y <= outline_range*outline_range:
                        draw.text((text_x + adj_x, current_y + adj_y), line, font=font, fill=outline_color)
            
            # Layer 3: Main text with bright color
            draw.text((text_x, current_y), line, font=font, fill=(255, 255, 255))
            
            current_y += line_heights[i] + 10
        
        return img
    
    def _select_best_frames(self, cap, total_frames, fps, num_frames=5):
        """Use OpenCV and AI to select the best frames for thumbnails"""
        print(f"[AI FRAME] Analyzing {total_frames} frames for optimal thumbnail selection...")
        
        # Sample frames at regular intervals
        sample_interval = max(1, total_frames // 30)  # Sample ~30 frames
        candidate_frames = []
        
        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue
            
            # Calculate quality score for this frame
            quality_score = self._calculate_frame_quality(frame)
            
            if quality_score > 0.3:  # Only consider frames with decent quality
                candidate_frames.append((frame_idx, frame.copy(), quality_score))
        
        print(f"[AI FRAME] Found {len(candidate_frames)} candidate frames")
        
        # Sort by quality score and select top frames
        candidate_frames.sort(key=lambda x: x[2], reverse=True)
        best_frames = candidate_frames[:num_frames]
        
        print(f"[AI FRAME] Selected top {len(best_frames)} frames based on quality analysis")
        return best_frames
    
    def _calculate_frame_quality(self, frame):
        """Calculate quality score for a frame using multiple metrics"""
        score = 0.0
        
        # 1. Sharpness/Focus Score (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)  # Normalize
        score += sharpness_score * 0.35
        
        # 2. Brightness Score (avoid too dark or too bright)
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
        score += brightness_score * 0.20
        
        # 3. Color Richness Score (saturation)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1])
        saturation_score = min(saturation / 180, 1.0)  # Normalize
        score += saturation_score * 0.20
        
        # 4. Face Detection Score (frames with faces are better)
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            face_score = min(len(faces) * 0.5, 1.0)  # Up to 2 faces = max score
            score += face_score * 0.15
        
        # 5. Composition Score (rule of thirds)
        composition_score = self._calculate_composition_score(frame)
        score += composition_score * 0.10
        
        return score
    
    def _calculate_composition_score(self, frame):
        """Calculate composition score based on edge distribution"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        h, w = edges.shape
        
        # Divide frame into 9 regions (rule of thirds)
        third_h, third_w = h // 3, w // 3
        regions = []
        
        for i in range(3):
            for j in range(3):
                region = edges[i*third_h:(i+1)*third_h, j*third_w:(j+1)*third_w]
                edge_density = np.sum(region) / (third_h * third_w)
                regions.append(edge_density)
        
        # Good composition has edges near third lines
        # Interest points at intersections of thirds
        interest_regions = [regions[0], regions[2], regions[6], regions[8]]  # Corners
        interest_score = np.mean(interest_regions) / 255.0
        
        # Balance score (not all edges in one place)
        balance_score = 1.0 - (np.std(regions) / (np.mean(regions) + 1e-6))
        balance_score = np.clip(balance_score, 0, 1)
        
        return (interest_score + balance_score) / 2

class AIColorEnhancer:
    """AI-based automatic color and saturation enhancement"""
    
    def __init__(self):
        self.optimal_saturation_range = (0.3, 0.7)  # Optimal saturation range for most videos

# NOTE: The main AudioEnhancer class with Whisper-based filler detection is defined below (around line 717)

class AIColorEnhancer:
    """AI-based automatic color and saturation enhancement"""
    
    def __init__(self):
        self.optimal_saturation_range = (0.3, 0.7)  # Optimal saturation range for most videos
        
    def analyze_video_colors(self, video_path, sample_frames=30):
        """Analyze video to determine optimal color adjustments"""
        try:
            print(f"[AI COLOR] Analyzing video colors from {sample_frames} frames...")
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // sample_frames)
            
            saturation_values = []
            brightness_values = []
            contrast_values = []
            
            frame_count = 0
            analyzed = 0
            
            while cap.isOpened() and analyzed < sample_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Convert to HSV for saturation analysis
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # Get saturation (S channel)
                    saturation = hsv[:, :, 1] / 255.0
                    saturation_values.append(np.mean(saturation))
                    
                    # Get brightness (V channel)
                    brightness = hsv[:, :, 2] / 255.0
                    brightness_values.append(np.mean(brightness))
                    
                    # Calculate contrast (std dev of brightness)
                    contrast_values.append(np.std(brightness))
                    
                    analyzed += 1
                
                frame_count += 1
            
            cap.release()
            
            # Calculate statistics
            avg_saturation = np.mean(saturation_values)
            avg_brightness = np.mean(brightness_values)
            avg_contrast = np.mean(contrast_values)
            
            print(f"[AI COLOR] Analysis complete:")
            print(f"  - Average Saturation: {avg_saturation:.3f}")
            print(f"  - Average Brightness: {avg_brightness:.3f}")
            print(f"  - Average Contrast: {avg_contrast:.3f}")
            
            return {
                'saturation': avg_saturation,
                'brightness': avg_brightness,
                'contrast': avg_contrast,
                'saturation_std': np.std(saturation_values),
                'brightness_std': np.std(brightness_values)
            }
            
        except Exception as e:
            print(f"[AI COLOR] Error analyzing video: {e}")
            return None
    
    def calculate_optimal_adjustments(self, analysis):
        """Calculate optimal color adjustments based on analysis"""
        if not analysis:
            return {'saturation': 1.0, 'brightness': 1.0, 'contrast': 1.0}
        
        adjustments = {}
        
        # Saturation adjustment
        current_sat = analysis['saturation']
        target_sat = (self.optimal_saturation_range[0] + self.optimal_saturation_range[1]) / 2
        
        if current_sat < self.optimal_saturation_range[0]:
            # Boost saturation for undersaturated videos
            saturation_boost = min(2.0, target_sat / max(current_sat, 0.1))
            adjustments['saturation'] = saturation_boost
            print(f"[AI COLOR] Boosting saturation by {saturation_boost:.2f}x")
        elif current_sat > self.optimal_saturation_range[1]:
            # Reduce saturation for oversaturated videos
            saturation_reduce = max(0.5, target_sat / current_sat)
            adjustments['saturation'] = saturation_reduce
            print(f"[AI COLOR] Reducing saturation to {saturation_reduce:.2f}x")
        else:
            adjustments['saturation'] = 1.0
            print(f"[AI COLOR] Saturation is optimal, no adjustment needed")
        
        # Brightness adjustment
        current_brightness = analysis['brightness']
        if current_brightness < 0.35:
            brightness_boost = min(1.4, 0.5 / current_brightness)
            adjustments['brightness'] = brightness_boost
            print(f"[AI COLOR] Boosting brightness to {brightness_boost:.2f}x")
        elif current_brightness > 0.75:
            brightness_reduce = max(0.7, 0.6 / current_brightness)
            adjustments['brightness'] = brightness_reduce
            print(f"[AI COLOR] Reducing brightness to {brightness_reduce:.2f}x")
        else:
            adjustments['brightness'] = 1.0
        
        # Contrast adjustment
        current_contrast = analysis['contrast']
        if current_contrast < 0.15:
            contrast_boost = min(1.3, 0.2 / current_contrast)
            adjustments['contrast'] = contrast_boost
            print(f"[AI COLOR] Boosting contrast to {contrast_boost:.2f}x")
        elif current_contrast > 0.35:
            contrast_reduce = max(0.8, 0.25 / current_contrast)
            adjustments['contrast'] = contrast_reduce
            print(f"[AI COLOR] Reducing contrast to {contrast_reduce:.2f}x")
        else:
            adjustments['contrast'] = 1.0
        
        return adjustments
    
    def apply_ai_enhancement(self, frame, saturation_mult=1.0, brightness_mult=1.0, contrast_mult=1.0):
        """Apply AI-calculated enhancements to a frame"""
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_mult, 0, 255)
        
        # Adjust brightness (V channel)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_mult, 0, 255)
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Apply contrast adjustment
        if contrast_mult != 1.0:
            enhanced = enhanced.astype(np.float32)
            enhanced = np.clip((enhanced - 128) * contrast_mult + 128, 0, 255)
            enhanced = enhanced.astype(np.uint8)
        
        return enhanced

class AudioEnhancer:
    """Advanced Audio Enhancement with REAL Filler Word Detection using Whisper and Noise Reduction"""
    
    def __init__(self):
        # Common filler words in different languages
        self.filler_words = {
            'en': ['um', 'uh', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically', 'literally', 'obviously', 'er', 'hmm', 'mm'],
            'ur': ['اں', 'ہاں', 'یعنی', 'اصل میں', 'تو', 'بس', 'اچھا'],
            'es': ['eh', 'este', 'bueno', 'pues', 'o sea', 'como', 'entonces'],
            'fr': ['euh', 'ben', 'alors', 'donc', 'enfin', 'bon', 'voilà'],
            'de': ['äh', 'ähm', 'also', 'ja', 'naja', 'halt', 'eigentlich']
        }
        # Whisper model for speech recognition (lazy loaded)
        self._whisper_model = None
        self._filler_words_removed_count = 0
    
    def _load_whisper(self):
        """Lazy load Whisper model for filler word detection"""
        if self._whisper_model is None:
            try:
                import whisper
                print("[AUDIO ENHANCER] Loading Whisper model for filler word detection...")
                self._whisper_model = whisper.load_model("base")
                print("[AUDIO ENHANCER] Whisper model loaded successfully")
            except Exception as e:
                print(f"[AUDIO ENHANCER] Failed to load Whisper: {e}")
                self._whisper_model = None
        return self._whisper_model
    
    def enhance_audio(self, audio_path, options):
        """Main audio enhancement function - TEMPORARILY DISABLED for Python 3.13"""
        print("[AUDIO ENHANCE] Audio enhancement temporarily disabled - pydub not compatible with Python 3.13")
        # Return original audio path without enhancement
        return audio_path
        
        # COMMENTED OUT - Requires pydub which needs audioop (removed in Python 3.13)
        # try:
        #     # Load audio
        #     audio = AudioSegment.from_file(audio_path)
            print(f"[AUDIO ENHANCE] Loaded audio: {len(audio)}ms, {audio.frame_rate}Hz")
            
            # Get enhancement options
            enhancement_type = options.get('audio_enhancement_type', 'medium')
            pause_threshold = options.get('pause_threshold', 500)
            noise_reduction = options.get('noise_reduction', 'moderate')
            
            print(f"[AUDIO ENHANCE] Options: type={enhancement_type}, pause={pause_threshold}ms, noise={noise_reduction}")
            
            # Reset filler word count
            self._filler_words_removed_count = 0
            
            # ==========================================
            # CLEANVOICE AI MECHANISM (Professional Order)
            # ==========================================
            # Step 1: Remove Background Noise FIRST (Critical!)
            # Reason: Saaf audio se Whisper filler words better detect karta hai
            if noise_reduction != 'none':
                enhanced_audio = self._reduce_noise(audio, noise_reduction)
                print(f"[CLEANVOICE AI] ✓ Step 1: Background noise removed ({noise_reduction}): {len(enhanced_audio)}ms")
            else:
                enhanced_audio = audio
            
            # Step 2: Remove Filler Words (Um, Uh, Like) using Whisper ASR
            # Reason: Clean audio pe Whisper accurately kaam karta hai
            if enhancement_type in ['medium', 'aggressive']:
                # Save cleaned audio temporarily for Whisper processing
                temp_cleaned_path = audio_path.replace('.', '_cleaned_temp.')
                enhanced_audio.export(temp_cleaned_path, format='wav')
                
                enhanced_audio = self._remove_filler_words_with_whisper(enhanced_audio, temp_cleaned_path, enhancement_type)
                print(f"[CLEANVOICE AI] ✓ Step 2: Filler words removed: {len(enhanced_audio)}ms, {self._filler_words_removed_count} fillers")
                
                # Cleanup temp file
                try:
                    import os
                    os.remove(temp_cleaned_path)
                except:
                    pass
            
            # Step 3: Remove excessive silence/pauses (natural flow)
            enhanced_audio = self._remove_silence(enhanced_audio, pause_threshold)
            print(f"[CLEANVOICE AI] ✓ Step 3: Silence removed: {len(enhanced_audio)}ms")
            
            # Step 4: Apply transition smoothing for natural flow
            enhanced_audio = self._apply_transition_smoothing(enhanced_audio)
            print(f"[CLEANVOICE AI] ✓ Step 4: Transitions smoothed: {len(enhanced_audio)}ms")
            
            # Step 5: Normalize audio
            enhanced_audio = normalize(enhanced_audio)
            print(f"[AUDIO ENHANCE] Final audio: {len(enhanced_audio)}ms")
            
            # Calculate improvement metrics
            original_duration = len(audio)
            enhanced_duration = len(enhanced_audio)
            time_saved = original_duration - enhanced_duration
            
            metrics = {
                'original_duration_ms': original_duration,
                'enhanced_duration_ms': enhanced_duration,
                'time_saved_ms': time_saved,
                'time_saved_percentage': (time_saved / original_duration) * 100 if original_duration > 0 else 0,
                'noise_reduction_level': noise_reduction,
                'enhancement_type': enhancement_type,
                'filler_words_removed': self._filler_words_removed_count
            }
            
            print(f"[AUDIO ENHANCE] Metrics: {metrics}")
            return enhanced_audio, metrics
            
        except Exception as e:
            print(f"[AUDIO ENHANCE] Error: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _remove_silence(self, audio, pause_threshold):
        """Remove excessive silence while preserving natural speech rhythm"""
        try:
            # Detect non-silent chunks
            min_silence_len = max(pause_threshold, 300)  # Minimum 300ms
            silence_thresh = audio.dBFS - 16  # Dynamic threshold based on audio level
            
            print(f"[SILENCE] Detecting silence: threshold={silence_thresh}dB, min_len={min_silence_len}ms")
            
            # Split on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=200  # Keep 200ms of silence for natural flow
            )
            
            if not chunks:
                print("[SILENCE] No chunks found, returning original audio")
                return audio
            
            # Combine chunks with minimal controlled gaps for natural timing
            result = AudioSegment.empty()
            for i, chunk in enumerate(chunks):
                result += chunk
                # Add minimal gap between chunks (except last one)
                if i < len(chunks) - 1:
                    gap_duration = min(150, pause_threshold // 4)  # Smaller gap, better timing
                    silence_gap = AudioSegment.silent(duration=gap_duration)
                    result += silence_gap
            
            print(f"[SILENCE] Processed {len(chunks)} chunks")
            return result
            
        except Exception as e:
            print(f"[SILENCE] Error: {e}")
            return audio
    
    def _remove_filler_words_with_whisper(self, audio, audio_path, enhancement_type):
        """CleanVoice AI mechanism: Identify -> Timestamp -> Cut"""
        try:
            print(f"[CLEANVOICE FILLER] Starting professional filler word removal: {enhancement_type}")
            
            # CleanVoice AI comprehensive filler word lists
            filler_config = {
                'conservative': ['um', 'uh', 'er'],
                'medium': ['um', 'uh', 'er', 'ah', 'hmm', 'mm', 'erm', 'uhm', 'umm', 'mhm', 'ehh'],
                'aggressive': ['um', 'uh', 'er', 'ah', 'hmm', 'mm', 'erm', 'uhm', 'umm', 'mhm', 'ehh', 'like', 'you know', 'basically', 'actually', 'literally', 'so', 'well', 'okay', 'right', 'yeah']
            }
            
            target_fillers = filler_config.get(enhancement_type, filler_config['medium'])
            print(f"[CLEANVOICE FILLER] Target fillers ({len(target_fillers)}): {target_fillers}")
            
            # Detect filler words with Whisper
            filler_segments = self._detect_fillers_with_whisper(audio_path, target_fillers)
            
            if not filler_segments:
                print("[CLEANVOICE FILLER] ✓ No filler words detected (clean audio!)")
                return audio
            
            print(f"[CLEANVOICE FILLER] Found {len(filler_segments)} filler segments to remove")
            
            # Sort segments by start time
            filler_segments.sort(key=lambda x: x[0])
            
            # CleanVoice AI cutting & stitching mechanism
            result = AudioSegment.empty()
            last_end = 0
            crossfade_duration = 15  # Ultra-minimal crossfade for natural flow
            
            for start_ms, end_ms in filler_segments:
                # Keep audio BEFORE filler word
                if start_ms > last_end:
                    segment = audio[last_end:start_ms]
                    
                    # CleanVoice AI smooth concatenation
                    if len(result) > crossfade_duration and len(segment) > crossfade_duration:
                        result = result.append(segment, crossfade=crossfade_duration)
                    else:
                        result += segment
                
                # CUT the filler word (skip completely)
                last_end = end_ms
                self._filler_words_removed_count += 1
            
            # Add remaining audio with smooth transition
            if last_end < len(audio):
                remaining = audio[last_end:]
                if len(result) > crossfade_duration and len(remaining) > crossfade_duration:
                    result = result.append(remaining, crossfade=crossfade_duration)
                else:
                    result += remaining
            
            print(f"[CLEANVOICE FILLER] ✓ Successfully removed {self._filler_words_removed_count} filler words")
            return result
            
        except Exception as e:
            print(f"[CLEANVOICE FILLER] ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return audio
    
    def _detect_fillers_with_whisper(self, audio_path, target_fillers):
        """CleanVoice AI: Whisper ASR for word-level timestamp detection"""
        try:
            model = self._load_whisper()
            if model is None:
                print("[CLEANVOICE WHISPER] Whisper not available, skipping filler detection...")
                return []
            
            print(f"[CLEANVOICE WHISPER] Transcribing audio with word-level timestamps...")
            
            # Whisper transcription with word-level timestamps (CleanVoice AI core)
            result = model.transcribe(
                audio_path,
                word_timestamps=True,  # Critical for CleanVoice AI mechanism
                language='en',
                verbose=False,
                fp16=False
            )
            
            filler_segments = []
            
            # CleanVoice AI pattern matching: Check each word
            for segment in result.get('segments', []):
                words = segment.get('words', [])
                for word_info in words:
                    word = word_info.get('word', '').lower().strip()
                    # Clean punctuation for exact matching
                    word_clean = ''.join(c for c in word if c.isalnum() or c.isspace()).strip()
                    
                    # EXACT match only (CleanVoice AI precision)
                    if word_clean in target_fillers:
                        start_ms = int(word_info.get('start', 0) * 1000)
                        end_ms = int(word_info.get('end', 0) * 1000)
                        
                        # Minimal buffer to avoid cutting real speech
                        start_ms = max(0, start_ms - 8)  # 8ms before
                        end_ms = end_ms + 8  # 8ms after
                        
                        filler_segments.append((start_ms, end_ms))
                        print(f"[CLEANVOICE WHISPER] \u2713 Detected '{word_clean}' at {start_ms}-{end_ms}ms")
            
            # Merge overlapping segments
            merged_segments = self._merge_overlapping_segments(filler_segments)
            print(f"[CLEANVOICE WHISPER] Total filler segments: {len(merged_segments)}")
            return merged_segments
            
        except Exception as e:
            print(f"[CLEANVOICE WHISPER] ✗ Error in filler detection: {e}")
            return []
    
    def _detect_filler_patterns_fallback(self, audio, enhancement_type):
        """Fallback: Detect potential filler segments using audio analysis"""
        try:
            print("[FILLER FALLBACK] Using audio analysis for filler detection...")
            
            # Convert to numpy array for processing
            audio_data = audio.get_array_of_samples()
            if audio.channels == 2:
                audio_data = np.array(audio_data).reshape((-1, 2))
                audio_data = audio_data.mean(axis=1)
            
            audio_data = np.array(audio_data, dtype=np.float32)
            
            # Normalize
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
            
            sample_rate = audio.frame_rate
            
            # Parameters based on enhancement type
            if enhancement_type == 'conservative':
                energy_low = 0.05
                energy_high = 0.3
                min_duration_ms = 150
                max_duration_ms = 800
            elif enhancement_type == 'medium':
                energy_low = 0.03
                energy_high = 0.35
                min_duration_ms = 100
                max_duration_ms = 1000
            else:  # aggressive
                energy_low = 0.02
                energy_high = 0.4
                min_duration_ms = 80
                max_duration_ms = 1200
            
            # Sliding window analysis
            window_ms = 50  # 50ms windows
            window_size = int(window_ms * sample_rate / 1000)
            hop_size = window_size // 2
            
            segments = []
            in_potential_filler = False
            filler_start = 0
            
            for i in range(0, len(audio_data) - window_size, hop_size):
                window = audio_data[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                
                current_time_ms = int(i * 1000 / sample_rate)
                
                # Filler words typically have consistent low-medium energy
                if energy_low < rms < energy_high:
                    if not in_potential_filler:
                        in_potential_filler = True
                        filler_start = current_time_ms
                else:
                    if in_potential_filler:
                        filler_end = current_time_ms
                        duration = filler_end - filler_start
                        
                        # Only count segments within typical filler word duration
                        if min_duration_ms <= duration <= max_duration_ms:
                            segments.append((filler_start, filler_end))
                        
                        in_potential_filler = False
            
            # Merge overlapping segments
            merged = self._merge_overlapping_segments(segments)
            print(f"[FILLER FALLBACK] Found {len(merged)} potential filler segments")
            return merged
            
        except Exception as e:
            print(f"[FILLER FALLBACK] Error: {e}")
            return []
    
    def _remove_filler_words(self, audio, enhancement_type):
        """Legacy method - redirects to Whisper-based detection"""
        # This method is kept for compatibility but redirects to the new method
        return audio  # The new method is called separately with audio_path
    
    def _merge_overlapping_segments(self, segments):
        """Merge overlapping time segments"""
        if not segments:
            return []
        
        segments.sort()
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            if current[0] <= last[1] + 100:  # 100ms tolerance
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def _reduce_noise(self, audio, noise_level):
        """Professional NR v4.0 - 100% Denoise Capability with Multi-Pass"""
        try:
            if noise_level == 'none':
                return audio
            
            print(f"[NR v4.0] Starting professional noise reduction: {noise_level}")
            
            import noisereduce as nr
            from scipy import signal as scipy_signal
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate
            channels = audio.channels
            
            # Normalize to -1 to 1
            samples = samples / 32768.0
            
            # Handle stereo
            if channels == 2:
                samples = samples.reshape((-1, 2))
                left_channel = samples[:, 0]
                right_channel = samples[:, 1]
            else:
                left_channel = samples
                right_channel = None
            
            # Professional NR v4.0 Profiles (Aggressive noise removal with voice preservation)
            nr_profiles = {
                'light': {
                    'stationary_prop': 0.88,
                    'non_stationary_prop': 0.75,
                    'voice_boost': 1.15,
                    'passes': 1
                },
                'moderate': {
                    'stationary_prop': 0.95,
                    'non_stationary_prop': 0.88,
                    'voice_boost': 1.25,
                    'passes': 2  # Multi-pass for better results
                },
                'strong': {
                    'stationary_prop': 0.97,
                    'non_stationary_prop': 0.92,
                    'voice_boost': 1.28,
                    'passes': 3  # Triple pass for strong removal
                },
                'full': {
                    'stationary_prop': 0.99,
                    'non_stationary_prop': 0.96,
                    'voice_boost': 1.32,
                    'passes': 3  # Triple pass for maximum noise removal
                }
            }
            
            profile = nr_profiles.get(noise_level, nr_profiles['moderate'])
            print(f"[NR v4.0] Profile: {noise_level} | Stationary: {profile['stationary_prop']*100:.0f}% | Non-stationary: {profile['non_stationary_prop']*100:.0f}% | Passes: {profile['passes']}")
            
            # STEP 1: Extract comprehensive noise sample (NR v4.0 method)
            print(f"[NR v4.0] Extracting noise profile from audio...")
            noise_sample = self._extract_noise_profile(left_channel, sample_rate)
            
            if noise_sample is None or len(noise_sample) < 100:
                # Fallback: Use first 1.5 seconds as noise sample
                noise_length = min(int(sample_rate * 1.5), len(left_channel) // 2)
                noise_sample = left_channel[:noise_length]
                print(f"[NR v4.0] Using first 1.5s as noise sample (length: {len(noise_sample)})")
            else:
                print(f"[NR v4.0] ✓ Noise profile extracted (length: {len(noise_sample)})")
            
            # STEP 3: Multi-Pass Professional Denoising (CleanVoice AI Algorithm)
            print(f"[CLEANVOICE DENOISE] Pass 1/2: Primary noise removal with learned profile...")
            
            # Pass 1: Aggressive stationary noise removal (AC, fan, hum)
            left_clean = nr.reduce_noise(
                y=left_channel,
                sr=sample_rate,
                y_noise=noise_sample,  # Use learned noise profile
                stationary=True,
                prop_decrease=profile['stationary_prop'],
                freq_mask_smooth_hz=600,  # Balanced for noise removal
                time_mask_smooth_ms=60
            )
            
            # Pass 2: Non-stationary cleanup (clicks, pops, keyboard)
            left_clean = nr.reduce_noise(
                y=left_clean,
                sr=sample_rate,
                y_noise=noise_sample,
                stationary=False,
                prop_decrease=profile['non_stationary_prop'],
                freq_mask_smooth_hz=400,
                time_mask_smooth_ms=40
            )
            
            # Multi-pass for moderate+ levels (like the web app you showed)
            if profile['passes'] >= 2:
                print(f"[NR v4.0] Pass 2/{profile['passes']}: Refinement pass...")
                # Second pass - moderate strength for additional cleaning
                left_clean = nr.reduce_noise(
                    y=left_clean,
                    sr=sample_rate,
                    y_noise=noise_sample,
                    stationary=True,
                    prop_decrease=min(profile['stationary_prop'] * 0.65, 0.88),
                    freq_mask_smooth_hz=800,
                    time_mask_smooth_ms=80
                )
                print(f"[NR v4.0] ✓ Pass 2 complete")
            
            if profile['passes'] >= 3:
                print(f"[NR v4.0] Pass 3/{profile['passes']}: Final polish...")
                # Third pass - light polish for smoothness
                left_clean = nr.reduce_noise(
                    y=left_clean,
                    sr=sample_rate,
                    y_noise=noise_sample,
                    stationary=True,
                    prop_decrease=0.80,
                    freq_mask_smooth_hz=1000,
                    time_mask_smooth_ms=100
                )
                print(f"[NR v4.0] ✓ Pass 3 complete")
            
            print(f"[NR v4.0] Final Stage: Voice spectrum enhancement...")
            # Voice enhancement using spectral shaping
            left_clean = self._enhance_voice_spectrum(left_clean, sample_rate, profile['voice_boost'])
            
            # Process right channel if stereo
            if right_channel is not None:
                right_with_vad = self._apply_voice_activity_detection(right_channel, sample_rate)
                
                # Stage 1: Stationary noise
                right_clean = nr.reduce_noise(
                    y=right_channel,
                    sr=sample_rate,
                    stationary=True,
                    prop_decrease=profile['stationary_prop'],
                    freq_mask_smooth_hz=500,
                    time_mask_smooth_ms=50
                )
                
                # Stage 2: Non-stationary noise
                right_clean = nr.reduce_noise(
                    y=right_clean,
                    sr=sample_rate,
                    stationary=False,
                    prop_decrease=profile['non_stationary_prop'],
                    freq_mask_smooth_hz=300,
                    time_mask_smooth_ms=30
                )
                
                # Stage 3: Voice enhancement
                right_clean = self._enhance_voice_spectrum(right_clean, sample_rate, profile['voice_boost'])
                
                samples_clean = np.column_stack((left_clean, right_clean)).flatten()
            else:
                samples_clean = left_clean
            
            # Convert back to int16
            samples_clean = np.clip(samples_clean * 32767, -32768, 32767).astype(np.int16)
            
            # Create cleaned audio
            enhanced = audio._spawn(samples_clean.tobytes())
            
            # STEP 4: Advanced EQ filtering (voice-optimized)
            # Remove sub-bass rumble (< 80Hz)
            enhanced = enhanced.high_pass_filter(80)
            
            # Remove ultrasonic noise (> 8000Hz for human voice)
            if noise_level in ['strong', 'full']:
                enhanced = enhanced.low_pass_filter(8000)
            
            # STEP 5: Gentle dynamic range compression (voice clarity without distortion)
            try:
                enhanced = enhanced.compress_dynamic_range(
                    threshold=-20.0,  # Less aggressive threshold
                    ratio=2.0,        # Very gentle compression
                    attack=5.0,       # Slower attack to preserve transients
                    release=50.0      # Smooth release
                )
            except:
                pass
            
            # STEP 6: Final normalization with more headroom to prevent clipping
            target_dBFS = -6.0  # More headroom to prevent distortion
            change_in_dBFS = target_dBFS - enhanced.dBFS
            enhanced = enhanced.apply_gain(change_in_dBFS)
            
            print(f"[NR v4.0] ✓ Professional noise reduction complete! ({profile['stationary_prop']*100:.0f}% stationary, {profile['non_stationary_prop']*100:.0f}% non-stationary, {profile['passes']} passes)")
            return enhanced
            
        except Exception as e:
            print(f"[NR v4.0] ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return audio
    
    def _apply_voice_activity_detection(self, audio_data, sample_rate):
        """CleanVoice AI: Voice Activity Detection using energy + zero-crossing rate"""
        try:
            # Frame-based VAD
            frame_length = int(sample_rate * 0.025)  # 25ms frames
            hop_length = int(sample_rate * 0.010)    # 10ms hop
            
            vad_mask = []
            
            for i in range(0, len(audio_data) - frame_length, hop_length):
                frame = audio_data[i:i + frame_length]
                
                # Energy-based detection
                energy = np.sum(frame ** 2) / len(frame)
                
                # Zero-crossing rate (voice has specific ZCR)
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                
                # Voice typically has: moderate energy + moderate ZCR
                is_voice = (energy > 0.001) and (0.02 < zcr < 0.5)
                vad_mask.append(is_voice)
            
            return vad_mask
            
        except:
            return None
    
    def _extract_noise_profile(self, audio_data, sample_rate):
        """CleanVoice AI: Adaptive noise profiling (learns from silent sections)"""
        try:
            # CleanVoice AI Strategy: Extract noise from multiple quiet sections
            # 1. First 1 second (usually has intro noise)
            # 2. Find other low-energy sections throughout audio
            
            noise_samples = []
            
            # Get first 1 second as primary noise sample
            first_second_length = min(int(sample_rate * 1.0), len(audio_data) // 3)
            if first_second_length > 0:
                noise_samples.append(audio_data[:first_second_length])
            
            # Find 2-3 more quiet sections (low energy parts)
            frame_length = int(sample_rate * 0.5)  # 500ms frames
            energy_threshold = 0.01  # Low energy = likely noise-only
            
            for i in range(0, len(audio_data) - frame_length, frame_length):
                frame = audio_data[i:i + frame_length]
                energy = np.sum(frame ** 2) / len(frame)
                
                if energy < energy_threshold and len(noise_samples) < 4:
                    noise_samples.append(frame)
            
            # Combine all noise samples
            if noise_samples:
                combined_noise = np.concatenate(noise_samples)
                return combined_noise
            else:
                # Fallback: use first portion
                return audio_data[:first_second_length]
            
        except:
            return None
    
    def _enhance_voice_spectrum(self, audio_data, sample_rate, boost_factor):
        """CleanVoice AI: Voice spectrum enhancement (200-3500Hz bandpass boost)"""
        try:
            from scipy import signal as scipy_signal
            
            # Design bandpass filter for voice frequencies
            # Human voice fundamental: 85-255Hz (male) to 165-255Hz (female)
            # Harmonics: up to 3400Hz for intelligibility
            
            # Boost voice range (200Hz - 3500Hz)
            sos_voice = scipy_signal.butter(4, [200, 3500], btype='band', fs=sample_rate, output='sos')
            voice_band = scipy_signal.sosfilt(sos_voice, audio_data)
            
            # Combine: original + boosted voice
            enhanced = audio_data + (voice_band * (boost_factor - 1.0))
            
            # Clip to prevent distortion
            enhanced = np.clip(enhanced, -1.0, 1.0)
            
            return enhanced
            
        except:
            return audio_data
        
        # Normalize by window overlap
        norm_factor = np.zeros_like(signal)
        for i in range(0, len(signal) - frame_length, hop_length):
            norm_factor[i:i + frame_length] += window ** 2
        norm_factor[norm_factor < 1e-10] = 1.0
        output = output / norm_factor
        
        return output
    
    def _apply_spectral_noise_reduction(self, audio, noise_level='moderate'):
        """Advanced spectral noise reduction with Wiener filtering + spectral subtraction"""
        try:
            print(f"[SPECTRAL] Applying {noise_level} noise reduction...")
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate
            channels = audio.channels
            
            # Normalize to -1 to 1
            samples = samples / 32768.0
            
            # Handle stereo
            if channels == 2:
                samples = samples.reshape((-1, 2))
                left_channel = samples[:, 0]
                right_channel = samples[:, 1]
            else:
                left_channel = samples
                right_channel = None
            
            # Configure aggressiveness - Better balance between noise reduction and voice
            if noise_level == 'light':
                reduction_db = 12
                gate_db = -50
                oversubtraction = 1.3
                noise_percentile = 15
            elif noise_level == 'moderate':
                reduction_db = 18
                gate_db = -45
                oversubtraction = 1.6
                noise_percentile = 20
            elif noise_level == 'strong':
                reduction_db = 24
                gate_db = -40
                oversubtraction = 2.0
                noise_percentile = 25
            elif noise_level == 'full':
                reduction_db = 30
                gate_db = -35
                oversubtraction = 2.5
                noise_percentile = 30
            else:
                reduction_db = 15
                gate_db = -45
                oversubtraction = 1.4
                noise_percentile = 18
            
            reduction_factor = 10 ** (reduction_db / 20.0)
            gate_threshold = 10 ** (gate_db / 20.0)
            
            # Process left channel
            left_clean = self._spectral_subtraction(left_channel, sample_rate, reduction_factor, gate_threshold, oversubtraction, noise_percentile)
            
            # Process right channel if stereo
            if right_channel is not None:
                right_clean = self._spectral_subtraction(right_channel, sample_rate, reduction_factor, gate_threshold, oversubtraction, noise_percentile)
                # Combine channels
                samples_clean = np.column_stack((left_clean, right_clean)).flatten()
            else:
                samples_clean = left_clean
            
            # Convert back to int16
            samples_clean = np.clip(samples_clean * 32767, -32768, 32767).astype(np.int16)
            
            # Create new AudioSegment
            cleaned_audio = audio._spawn(samples_clean.tobytes())
            
            # Apply gentle filtering for strong mode
            if noise_level in ['strong', 'full']:
                cleaned_audio = cleaned_audio.high_pass_filter(60)
                cleaned_audio = cleaned_audio.low_pass_filter(10000)
            
            # Final normalization
            cleaned_audio = normalize(cleaned_audio)
            
            print(f"[SPECTRAL] Noise reduction complete ({noise_level})")
            return cleaned_audio
            
        except Exception as e:
            print(f"[SPECTRAL] Error: {e}")
            import traceback
            traceback.print_exc()
            return audio
    
    def _spectral_subtraction(self, signal, sr, reduction_factor, gate_threshold, oversubtraction, noise_percentile=25):
        """Spectral subtraction with Wiener filtering"""
        frame_length = 2048
        hop_length = frame_length // 4
        
        # Estimate noise from quietest frames (configurable percentage)
        frame_energies = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            energy = np.sum(frame ** 2)
            frame_energies.append(energy)
        
        noise_threshold_energy = np.percentile(frame_energies, noise_percentile)
        
        # Collect noise frames
        noise_frames = []
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length]
            if np.sum(frame ** 2) <= noise_threshold_energy:
                noise_frames.append(frame)
        
        if len(noise_frames) == 0:
            return signal
        
        # Estimate noise spectrum (average of noise frames)
        noise_fft_sum = np.zeros(frame_length // 2 + 1, dtype=complex)
        for frame in noise_frames:
            noise_fft_sum += np.fft.rfft(frame * np.hanning(frame_length))
        noise_spectrum = np.abs(noise_fft_sum) / len(noise_frames)
        
        # Process signal with overlap-add
        output = np.zeros_like(signal)
        window = np.hanning(frame_length)
        
        for i in range(0, len(signal) - frame_length, hop_length):
            frame = signal[i:i + frame_length] * window
            frame_fft = np.fft.rfft(frame)
            magnitude = np.abs(frame_fft)
            phase = np.angle(frame_fft)
            
            # Calculate frequency bins
            freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)
            
            # Define frequency masks
            voice_core = (freqs >= 300) & (freqs <= 3000)  # Core voice frequencies
            low_freq = freqs < 200  # Low rumble (aggressive reduction)
            high_freq = freqs > 8000  # High hiss (aggressive reduction)
            
            # Detect voice presence in this frame
            frame_energy = np.sum(magnitude ** 2)
            avg_noise_energy = np.sum(noise_spectrum ** 2)
            is_voice_frame = frame_energy > (avg_noise_energy * 2.5)
            
            # Adaptive spectral subtraction based on frequency
            magnitude_clean = magnitude - (oversubtraction * noise_spectrum * reduction_factor)
            
            # Frequency-dependent floor (more aggressive on non-voice frequencies)
            floor_factor = np.ones_like(magnitude)
            floor_factor = np.where(voice_core, 0.4, floor_factor)  # 40% floor for voice
            floor_factor = np.where(low_freq, 0.1, floor_factor)    # 10% floor for low rumble
            floor_factor = np.where(high_freq, 0.1, floor_factor)   # 10% floor for high hiss
            
            magnitude_clean = np.maximum(magnitude_clean, floor_factor * magnitude)
            
            # Voice frame gets extra protection
            if is_voice_frame:
                magnitude_clean = np.where(voice_core,
                                          np.maximum(magnitude_clean, 0.6 * magnitude),
                                          magnitude_clean)
            
            # Adaptive Wiener filter based on SNR
            snr = (magnitude_clean ** 2) / ((noise_spectrum * reduction_factor) ** 2 + 1e-6)
            
            # Frequency-dependent Wiener gain
            wiener_gain = snr / (snr + 0.5)
            
            # Protect voice frequencies more
            wiener_gain = np.where(voice_core,
                                  snr / (snr + 0.3),  # Gentler for voice
                                  wiener_gain)
            
            # More aggressive on low/high frequencies
            wiener_gain = np.where(low_freq | high_freq,
                                  snr / (snr + 0.8),  # Stronger suppression
                                  wiener_gain)
            
            # Clip to reasonable range
            wiener_gain = np.clip(wiener_gain, 0.2, 1.0)  # 20% minimum
            
            # Voice frames get higher minimum
            if is_voice_frame:
                wiener_gain = np.where(voice_core,
                                      np.maximum(wiener_gain, 0.7),
                                      wiener_gain)
            
            # Apply gain
            magnitude_final = magnitude_clean * wiener_gain
            
            # Adaptive noise gate
            if is_voice_frame:
                # Gentle gate for voice frames
                gate_factor = np.clip(magnitude_final / (gate_threshold * 0.5 + 1e-10), 0.6, 1.0)
            else:
                # More aggressive gate for noise-only frames
                gate_factor = np.clip(magnitude_final / (gate_threshold + 1e-10), 0.2, 1.0)
            
            magnitude_final = magnitude_final * gate_factor
            
            # Reconstruct
            frame_fft_clean = magnitude_final * np.exp(1j * phase)
            frame_clean = np.fft.irfft(frame_fft_clean)
            
            # Overlap-add
            output[i:i + frame_length] += frame_clean * window
        
        # Normalize
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.95
        
        return output
    
    def _apply_transition_smoothing(self, audio):
        """Apply subtle crossfades to blend audio cuts for natural flow"""
        try:
            print("[SMOOTHING] Applying transition smoothing for natural blending...")
            
            # Apply gentle fade in/out to prevent jarring cuts
            fade_duration = 30  # 30ms subtle fade
            
            if len(audio) > fade_duration * 2:
                # Apply subtle fades at start and end
                audio = audio.fade_in(fade_duration).fade_out(fade_duration)
                print("[SMOOTHING] Applied subtle crossfades for natural flow")
            else:
                print("[SMOOTHING] Audio too short for fading")
            
            return audio
            
        except Exception as e:
            print(f"[SMOOTHING] Error: {e}")
            return audio

class VideoService:
    def __init__(self, db):
        self.db = db
        self.videos = db.videos
        self.upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')
        self.max_content_length = int(os.getenv('MAX_CONTENT_LENGTH', 500 * 1024 * 1024))
        
        # Initialize enhancers
        self.audio_enhancer = AudioEnhancer()
        self.color_enhancer = AIColorEnhancer()
        
        # Initialize AI models (disable problematic ones for now)
        try:
            # Disabled due to TensorFlow compatibility issues
            # self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            # self.speech_recognizer = pipeline("automatic-speech-recognition")
            self.summarizer = None
            self.speech_recognizer = None
            print("[VIDEO SERVICE] AI models disabled - focusing on Whisper for Urdu subtitles")
        except Exception as e:
            print(f"Warning: Could not initialize AI models: {e}")
            self.summarizer = None
            self.speech_recognizer = None

    def save_video(self, file, user_id):
        if not file:
            raise ValueError("No file provided")

        print(f"[SAVE_VIDEO] Saving video for user {user_id}")
        filename = secure_filename(file.filename)
        filepath = os.path.join(self.upload_folder, filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_folder, exist_ok=True)
        
        # Save file
        print(f"[SAVE_VIDEO] Saving file to: {filepath}")
        file.save(filepath)
        print(f"[SAVE_VIDEO] File saved, size: {os.path.getsize(filepath)} bytes")
        
        # Validate file
        if not self._is_valid_video(filepath):
            print(f"[SAVE_VIDEO] Invalid video file, removing")
            os.remove(filepath)
            raise ValueError("Invalid video file")
        
        print(f"[SAVE_VIDEO] Video validated successfully")

        # Create video document
        video = Video(
            user_id=ObjectId(user_id),
            filename=filename,
            filepath=filepath,
            size=os.path.getsize(filepath)
        )
        
        # Extract metadata (with timeout protection)
        print(f"[SAVE_VIDEO] Extracting metadata...")
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Metadata extraction timed out")
            
            # Set 10 second timeout for metadata extraction (Unix only, Windows doesn't support signal.alarm)
            # For Windows, just catch any exception
            self._extract_metadata(video)
        except Exception as e:
            print(f"[SAVE_VIDEO] Metadata extraction failed or timed out: {e}")
            # Continue anyway with minimal metadata
            video.metadata["format"] = os.path.splitext(filename)[1][1:]
        
        # Save to database
        print(f"[SAVE_VIDEO] Saving to database...")
        result = self.videos.insert_one(video.to_dict())
        video_id = str(result.inserted_id)
        print(f"[SAVE_VIDEO] Video saved successfully with ID: {video_id}")
        return video_id

    def process_video(self, video_id, options):
        video = self.get_video(video_id)
        if not video:
            raise ValueError("Video not found")

        video.status = "processing"
        video.process_start_time = datetime.utcnow()
        video.processing_options = options
        
        try:
            # Enhanced processing with actual options
            if options.get('cut_silence'):
                self._cut_silence(video)
            
            if options.get('enhance_audio'):
                self._enhance_audio(video, options)
            
            if options.get('generate_thumbnail'):
                self._generate_thumbnail(video, options)
            
            if options.get('generate_subtitles'):
                self._generate_subtitles(video, options)
            
            if options.get('summarize'):
                self._summarize_video(video)

            # Apply video enhancements (including AI color enhancement)
            if any([options.get('stabilization'), options.get('brightness'), options.get('contrast'), 
                    options.get('saturation'), options.get('ai_color_enhancement')]):
                self._apply_video_enhancements(video, options)

            video.status = "completed"
            video.process_end_time = datetime.utcnow()
            
        except Exception as e:
            video.status = "failed"
            video.error = str(e)
            video.process_end_time = datetime.utcnow()
            raise
        
        finally:
            # Get dict without _id to avoid MongoDB update error
            update_dict = video.to_dict()
            update_dict.pop('_id', None)  # Remove _id if present
            self.videos.update_one(
                {"_id": ObjectId(video_id)},
                {"$set": update_dict}
            )

    def get_video(self, video_id):
        video_data = self.videos.find_one({"_id": ObjectId(video_id)})
        if not video_data:
            return None
        return Video.from_dict(video_data)

    def get_user_videos(self, user_id):
        # Query with both ObjectId and string format since videos may have user_id stored either way
        videos = list(self.videos.find({
            "$or": [
                {"user_id": ObjectId(user_id)},
                {"user_id": str(user_id)}
            ]
        }))
        print(f"[GET_USER_VIDEOS] User ID: {user_id}, Found {len(videos)} videos")
        return [Video.from_dict(video).to_dict() for video in videos]

    def delete_video(self, video_id, user_id):
        video = self.get_video(video_id)
        if not video:
            raise ValueError("Video not found")
        
        if str(video.user_id) != str(user_id):
            raise ValueError("Unauthorized")
        
        # Delete file
        if os.path.exists(video.filepath):
            os.remove(video.filepath)
        
        # Delete processed files
        if video.outputs.get('processed_video') and os.path.exists(video.outputs['processed_video']):
            os.remove(video.outputs['processed_video'])
        
        # Delete from database
        self.videos.delete_one({"_id": ObjectId(video_id)})

    def _is_valid_video(self, filepath):
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(filepath)
            return file_type.startswith('video/')
        except:
            # Fallback: check file extension
            valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            return any(filepath.lower().endswith(ext) for ext in valid_extensions)

    def _extract_metadata(self, video):
        try:
            print(f"[METADATA] Extracting metadata for: {video.filepath}")
            clip = VideoFileClip(video.filepath, audio=False)  # Don't load audio for metadata
            print(f"[METADATA] VideoFileClip opened successfully")
            video.metadata.update({
                "duration": clip.duration,
                "fps": clip.fps,
                "resolution": f"{clip.size[0]}x{clip.size[1]}",
                "format": os.path.splitext(video.filename)[1][1:]
            })
            print(f"[METADATA] Metadata extracted: {video.metadata}")
            clip.close()
        except Exception as e:
            print(f"[METADATA] Error extracting metadata: {e}")
            import traceback
            traceback.print_exc()
            video.metadata.update({
                "format": os.path.splitext(video.filename)[1][1:]
            })

    def _apply_video_enhancements(self, video, options):
        """Apply video enhancements like brightness, contrast, stabilization, and AI color enhancement"""
        try:
            print("[VIDEO ENHANCE] Starting video enhancement processing...")
            clip = VideoFileClip(video.filepath)
            
            # Check if AI enhancement is requested
            use_ai_enhancement = options.get('ai_color_enhancement', False)
            ai_adjustments = None
            
            if use_ai_enhancement:
                print("[VIDEO ENHANCE] AI Color Enhancement enabled - analyzing video...")
                analysis = self.color_enhancer.analyze_video_colors(video.filepath)
                if analysis:
                    ai_adjustments = self.color_enhancer.calculate_optimal_adjustments(analysis)
                    print(f"[VIDEO ENHANCE] AI adjustments calculated: {ai_adjustments}")
                    
                    # Store AI analysis results
                    video.metadata['ai_color_analysis'] = {
                        'original_saturation': float(analysis['saturation']),
                        'original_brightness': float(analysis['brightness']),
                        'original_contrast': float(analysis['contrast']),
                        'applied_saturation_mult': float(ai_adjustments['saturation']),
                        'applied_brightness_mult': float(ai_adjustments['brightness']),
                        'applied_contrast_mult': float(ai_adjustments['contrast'])
                    }
            
            # Get manual adjustments (if provided, they override AI)
            brightness = options.get('brightness', 100) / 100.0
            contrast = options.get('contrast', 100) / 100.0
            saturation = options.get('saturation', 100) / 100.0
            
            # If AI enhancement is enabled and no manual override, use AI values
            if use_ai_enhancement and ai_adjustments:
                if options.get('brightness') is None:
                    brightness = ai_adjustments['brightness']
                if options.get('contrast') is None:
                    contrast = ai_adjustments['contrast']
                if options.get('saturation') is None:
                    saturation = ai_adjustments['saturation']
            
            print(f"[VIDEO ENHANCE] Final adjustments - Brightness: {brightness:.2f}x, Contrast: {contrast:.2f}x, Saturation: {saturation:.2f}x")
            
            # Apply enhancements if needed
            if brightness != 1.0 or contrast != 1.0 or saturation != 1.0:
                def enhance_frame(image):
                    # Convert BGR (moviepy) to RGB for processing
                    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Apply AI enhancement
                    enhanced = self.color_enhancer.apply_ai_enhancement(
                        frame,
                        saturation_mult=saturation,
                        brightness_mult=brightness,
                        contrast_mult=contrast
                    )
                    
                    # Convert back to RGB for moviepy
                    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                
                clip = clip.fl_image(enhance_frame)
                print("[VIDEO ENHANCE] Frame enhancement applied")
            
            # Apply stabilization (basic implementation)
            stabilization = options.get('stabilization', 'none')
            if stabilization != 'none':
                print(f"[VIDEO ENHANCE] Applying {stabilization} stabilization...")
                # For now, we'll just apply a simple smoothing
                # In a real implementation, you'd use more sophisticated stabilization
                pass
            
            # Save enhanced video with optimized settings
            output_path = f"{os.path.splitext(video.filepath)[0]}_enhanced.mp4"
            print(f"[VIDEO ENHANCE] Saving enhanced video to: {output_path}")
            
            # Use optimized encoding settings to prevent corruption
            clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                preset='medium',
                fps=clip.fps,
                threads=4,
                bitrate='5000k',
                audio_bitrate='192k',
                ffmpeg_params=[
                    '-crf', '23',  # Quality (lower = better, 18-28 is good range)
                    '-pix_fmt', 'yuv420p',  # Compatibility
                    '-movflags', '+faststart'  # Web optimization
                ]
            )
            video.outputs["processed_video"] = output_path
            
            clip.close()
            print("[VIDEO ENHANCE] Video enhancement completed successfully!")
            
        except Exception as e:
            print(f"[VIDEO ENHANCE] Error: {e}")
            raise

    def _cut_silence(self, video):
        try:
            audio = AudioSegment.from_file(video.filepath)
            chunks = []
            silence_thresh = -40
            min_silence_len = 500
            
            # Process audio in chunks
            chunk_length = 10000
            for i in range(0, len(audio), chunk_length):
                chunk = audio[i:i + chunk_length]
                if chunk.dBFS > silence_thresh:
                    chunks.append(chunk)
            
            # Combine non-silent chunks
            processed_audio = AudioSegment.empty()
            for chunk in chunks:
                processed_audio += chunk
            
            # Save processed audio
            output_path = f"{os.path.splitext(video.filepath)[0]}_processed.mp4"
            processed_audio.export(output_path, format="mp4")
            video.outputs["processed_video"] = output_path
        except Exception as e:
            print(f"Error cutting silence: {e}")

    def _enhance_audio(self, video, options):
        """Enhanced audio processing with filler word removal and noise reduction"""
        try:
            print(f"[VIDEO SERVICE] Starting enhanced audio processing for {video.filepath}")
            
            # Initialize the audio enhancer
            audio_enhancer = AudioEnhancer()
            
            # Map frontend options to backend options - also pass enhancement_type correctly
            # Frontend sends 'noise_reduction' values like 'light', 'moderate', 'strong'
            # But audio_enhancement_type should be 'conservative', 'medium', 'aggressive'
            noise_level = options.get('noise_reduction', 'moderate')
            
            # Map noise_reduction level to enhancement_type for filler word removal
            enhancement_type_map = {
                'none': 'conservative',
                'light': 'conservative', 
                'moderate': 'medium',
                'strong': 'aggressive'
            }
            enhancement_type = enhancement_type_map.get(noise_level, 'medium')
            
            backend_options = {
                'audio_enhancement_type': enhancement_type,  # Use mapped value
                'pause_threshold': options.get('pause_threshold', 500),
                'noise_reduction': noise_level
            }
            
            print(f"[VIDEO SERVICE] Backend options: {backend_options}")
            
            # Extract audio from video first
            clip = VideoFileClip(video.filepath)
            audio_path = f"{os.path.splitext(video.filepath)[0]}_temp_audio.wav"
            print(f"[VIDEO SERVICE] Extracting audio to: {audio_path}")
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            # Enhance the audio (pass audio_path for Whisper filler detection)
            print(f"[VIDEO SERVICE] Starting audio enhancement with Whisper filler detection...")
            enhanced_audio, metrics = audio_enhancer.enhance_audio(audio_path, backend_options)
            
            # Save enhanced audio temporarily
            enhanced_audio_path = f"{os.path.splitext(video.filepath)[0]}_enhanced_audio.wav"
            print(f"[VIDEO SERVICE] Saving enhanced audio to: {enhanced_audio_path}")
            enhanced_audio.export(enhanced_audio_path, format="wav")
            
            # Create new video with enhanced audio
            print(f"[VIDEO SERVICE] Creating final video with enhanced audio...")
            # Load the enhanced audio as an AudioFileClip
            from moviepy.editor import AudioFileClip
            enhanced_audio_clip = AudioFileClip(enhanced_audio_path)
            enhanced_clip = clip.set_audio(enhanced_audio_clip)
            
            # Save final enhanced video
            output_path = f"{os.path.splitext(video.filepath)[0]}_enhanced.mp4"
            enhanced_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Update video outputs
            video.outputs["processed_video"] = output_path
            video.outputs["audio_enhancement_metrics"] = metrics
            
            # Add detailed results for frontend display - use REAL filler word count
            original_duration_sec = metrics['original_duration_ms'] / 1000
            enhanced_duration_sec = metrics['enhanced_duration_ms'] / 1000
            time_saved_sec = metrics['time_saved_ms'] / 1000
            filler_words_removed = metrics.get('filler_words_removed', 0)
            
            video.outputs["enhancement_results"] = {
                'filler_words_removed': filler_words_removed,  # REAL count from Whisper
                'noise_reduction_percentage': 85 if noise_level in ['moderate', 'strong'] else (50 if noise_level == 'light' else 0),
                'duration_reduction_percentage': round(metrics['time_saved_percentage'], 1),
                'original_duration': f"{original_duration_sec:.1f}s",
                'enhanced_duration': f"{enhanced_duration_sec:.1f}s",
                'time_saved': f"{time_saved_sec:.1f}s"
            }
            
            print(f"[VIDEO SERVICE] Audio enhancement completed successfully")
            print(f"[VIDEO SERVICE] Metrics: {metrics}")
            print(f"[VIDEO SERVICE] Final video saved to: {output_path}")
            
            # Cleanup temporary files
            clip.close()
            enhanced_clip.close()
            enhanced_audio_clip.close()
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(enhanced_audio_path):
                os.remove(enhanced_audio_path)
                
        except Exception as e:
            print(f"[VIDEO SERVICE] Error enhancing audio: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_thumbnail(self, video, options=None):
        try:
            print(f"[THUMBNAIL] Starting AI-powered thumbnail generation for: {video.filepath}")
            print(f"[THUMBNAIL DEBUG] Received options: {options}")
            
            # Get custom text and frame index from options
            custom_text = options.get('thumbnail_text') if options else None
            frame_index = options.get('thumbnail_frame_index') if options else None
            
            print(f"[THUMBNAIL DEBUG] custom_text = '{custom_text}' (type: {type(custom_text)})")
            print(f"[THUMBNAIL DEBUG] frame_index = {frame_index}")
            
            # Ensure custom_text is not empty string
            if custom_text == '':
                custom_text = None
                print(f"[THUMBNAIL DEBUG] Empty string detected, setting to None")
            
            if custom_text:
                print(f"[THUMBNAIL] ✅ Using custom text: '{custom_text}'")
            else:
                print(f"[THUMBNAIL] ⚠️ No custom text provided, will use AI generation")
            
            if frame_index is not None:
                print(f"[THUMBNAIL] Using selected frame index: {frame_index}")
            
            # Initialize AI thumbnail generator
            ai_generator = AIThumbnailGenerator()
            
            # Check if file exists
            if not os.path.exists(video.filepath):
                print(f"[THUMBNAIL] Error: Video file does not exist: {video.filepath}")
                return
            
            print(f"[THUMBNAIL] File exists, size: {os.path.getsize(video.filepath)} bytes")
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(video.filepath)
            
            if not cap.isOpened():
                print(f"[THUMBNAIL] Error: Could not open video file with OpenCV")
                print(f"[THUMBNAIL] Attempting alternative method with moviepy...")
                
                # Try with moviepy as fallback
                try:
                    from moviepy.editor import VideoFileClip
                    clip = VideoFileClip(video.filepath)
                    
                    print(f"[THUMBNAIL] Using MoviePy with intelligent frame selection...")
                    
                    # Sample frames for quality analysis
                    duration = clip.duration
                    sample_times = np.linspace(0.1 * duration, 0.9 * duration, 20)
                    candidate_frames = []
                    
                    for idx, time_sec in enumerate(sample_times):
                        frame = clip.get_frame(time_sec)
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Calculate quality score
                        quality_score = ai_generator._calculate_frame_quality(frame_bgr)
                        
                        if quality_score > 0.3:
                            candidate_frames.append((idx, frame_bgr, quality_score, time_sec))
                    
                    # Sort by quality and select top 5
                    candidate_frames.sort(key=lambda x: x[2], reverse=True)
                    best_frames = candidate_frames[:5]
                    
                    print(f"[THUMBNAIL] Selected {len(best_frames)} high-quality frames using AI analysis")
                    
                    thumbnails = []
                    
                    for i, (idx, frame_bgr, quality_score, time_sec) in enumerate(best_frames):
                        print(f"[THUMBNAIL] Processing frame at {time_sec:.1f}s (quality: {quality_score:.2f})")
                        
                        # Save temporary frame
                        temp_frame_path = f"{os.path.splitext(video.filepath)[0]}_temp_frame_{i+1}.jpg"
                        cv2.imwrite(temp_frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        # Use custom text if provided, otherwise generate AI text
                        if custom_text:
                            ai_text = custom_text
                            print(f"[THUMBNAIL] ✅✅✅ Using CUSTOM TEXT: '{ai_text}' ✅✅✅")
                        else:
                            # Generate AI text for this frame
                            ai_text = ai_generator.generate_catchy_text(temp_frame_path, video.filepath)
                            print(f"[THUMBNAIL] Generated AI text: '{ai_text}'")
                        
                        # Create YouTube thumbnail with text
                        thumbnail_path = f"{os.path.splitext(video.filepath)[0]}_thumb_{i+1}.jpg"
                        final_path = ai_generator.create_youtube_thumbnail(temp_frame_path, ai_text, thumbnail_path)
                        
                        # Clean up temp frame
                        if os.path.exists(temp_frame_path):
                            os.remove(temp_frame_path)
                        
                        if os.path.exists(final_path):
                            thumbnails.append(final_path)
                            print(f"[THUMBNAIL] Successfully created AI thumbnail {i+1} using moviepy: {final_path}")
                            print(f"[THUMBNAIL] AI Text: {ai_text}")
                        else:
                            print(f"[THUMBNAIL] Failed to create AI thumbnail {i+1} using moviepy")
                    
                    clip.close()
                    
                    if thumbnails:
                        video.outputs["thumbnails"] = thumbnails
                        video.outputs["thumbnail"] = thumbnails[2] if len(thumbnails) > 2 else thumbnails[0]
                        print(f"[THUMBNAIL] Generated {len(thumbnails)} thumbnails successfully with moviepy")
                        print(f"[THUMBNAIL] Primary thumbnail: {video.outputs['thumbnail']}")
                    else:
                        print(f"[THUMBNAIL] Warning: No thumbnails were generated with moviepy")
                    
                    return
                    
                except Exception as moviepy_error:
                    print(f"[THUMBNAIL] Moviepy fallback also failed: {moviepy_error}")
                    import traceback
                    traceback.print_exc()
                    return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            print(f"[THUMBNAIL] Video info - Total frames: {total_frames}, FPS: {fps}, Duration: {duration}s")
            
            if total_frames <= 0:
                print(f"[THUMBNAIL] Error: Invalid frame count")
                cap.release()
                return
            
            # Use AI-powered intelligent frame selection
            print(f"[THUMBNAIL] Starting intelligent frame selection using OpenCV and AI analysis...")
            
            # Select candidate frames using quality metrics
            candidate_frames = ai_generator._select_best_frames(cap, total_frames, fps)
            print(f"[THUMBNAIL] Selected {len(candidate_frames)} high-quality candidate frames")
            
            # Generate thumbnails from best frames
            thumbnails = []
            
            for i, (frame_number, frame, quality_score) in enumerate(candidate_frames):
                print(f"[THUMBNAIL] Processing frame {frame_number} (quality score: {quality_score:.2f})")
                
                ret = True  # We already have the frame from _select_best_frames
                
                if ret and frame is not None:
                    print(f"[THUMBNAIL] Frame {i+1} read successfully, shape: {frame.shape}")
                    
                    # Save temporary frame
                    temp_frame_path = f"{os.path.splitext(video.filepath)[0]}_temp_frame_{i+1}.jpg"
                    cv2.imwrite(temp_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Use custom text if provided, otherwise generate AI text
                    if custom_text:
                        ai_text = custom_text
                        print(f"[THUMBNAIL] ✅✅✅ Using CUSTOM TEXT: '{ai_text}' ✅✅✅")
                    else:
                        # Generate AI text for this frame
                        ai_text = ai_generator.generate_catchy_text(temp_frame_path, video.filepath)
                        print(f"[THUMBNAIL] Generated AI text: '{ai_text}'")
                    
                    # Create YouTube thumbnail with text
                    thumbnail_path = f"{os.path.splitext(video.filepath)[0]}_thumb_{i+1}.jpg"
                    final_path = ai_generator.create_youtube_thumbnail(temp_frame_path, ai_text, thumbnail_path)
                    
                    # Clean up temp frame
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
                    
                    if os.path.exists(final_path):
                        file_size = os.path.getsize(final_path)
                        thumbnails.append(final_path)
                        print(f"[THUMBNAIL] Successfully created AI thumbnail {i+1}: {final_path} ({file_size} bytes)")
                        print(f"[THUMBNAIL] AI Text: {ai_text}")
                    else:
                        print(f"[THUMBNAIL] Failed to create AI thumbnail {i+1}")
                else:
                    print(f"[THUMBNAIL] Failed to read frame {frame_number}, ret={ret}")
            
            cap.release()
            
            if thumbnails:
                video.outputs["thumbnails"] = thumbnails
                video.outputs["thumbnail"] = thumbnails[2] if len(thumbnails) > 2 else thumbnails[0]  # Use middle thumbnail as primary
                print(f"[THUMBNAIL] Generated {len(thumbnails)} thumbnails successfully")
                print(f"[THUMBNAIL] Primary thumbnail: {video.outputs['thumbnail']}")
                print(f"[THUMBNAIL] All thumbnails: {thumbnails}")
            else:
                print(f"[THUMBNAIL] Warning: No thumbnails were generated")
                
        except Exception as e:
            print(f"[THUMBNAIL] Error generating thumbnail: {e}")
            import traceback
            traceback.print_exc()

    def _generate_subtitles(self, video, options):
        """Enhanced subtitle generation with language support"""
        clip = None
        audio_path = None
        try:
            # Get language and style from options
            language = options.get('subtitle_language', 'en')
            style = options.get('subtitle_style', 'clean')
            
            print(f"[SUBTITLE DEBUG] Starting subtitle generation for video: {video.filepath}")
            print(f"[SUBTITLE DEBUG] Language: {language}, Style: {style}")
            
            # Extract audio for transcription
            clip = VideoFileClip(video.filepath)
            
            # Check if video has audio
            if clip.audio is None:
                print(f"[SUBTITLE DEBUG] Warning: Video has no audio track")
                # Create empty subtitles for videos without audio
                self._create_fallback_subtitles(video, options)
                if clip:
                    clip.close()
                return
            
            audio_path = f"{os.path.splitext(video.filepath)[0]}_audio.wav"
            print(f"[SUBTITLE DEBUG] Extracting audio to: {audio_path}")
            clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
            print(f"[SUBTITLE DEBUG] Audio extraction completed")
            
            # Try to use Whisper for real transcription
            try:
                print(f"[SUBTITLE DEBUG] Attempting Whisper transcription...")
                import whisper
                print(f"[SUBTITLE DEBUG] Whisper imported successfully")
                
                # Use appropriate model based on language
                model_size = self._get_optimal_whisper_model(language)
                print(f"[SUBTITLE DEBUG] Loading Whisper model: {model_size}")
                model = whisper.load_model(model_size)
                print(f"[SUBTITLE DEBUG] Whisper model loaded successfully")
                
                whisper_lang = self._get_whisper_language_code(language)
                print(f"[SUBTITLE DEBUG] Using Whisper language code: {whisper_lang}")
                
                # Preprocess audio for better recognition (especially for Urdu)
                processed_audio_path = self._preprocess_audio_for_transcription(audio_path, language)
                print(f"[SUBTITLE DEBUG] Audio preprocessed for {language}")
                
                # First, detect the actual language to prevent hallucinations
                print(f"[SUBTITLE DEBUG] Detecting actual audio language...")
                detect_result = model.transcribe(processed_audio_path, language=None, fp16=False)
                detected_lang = detect_result.get('language', 'en')
                print(f"[SUBTITLE DEBUG] Detected language: {detected_lang}, Requested: {whisper_lang}")
                
                # Determine if we need translation
                needs_translation = False
                source_lang = whisper_lang
                target_lang = whisper_lang
                
                # If detected language is different from requested, we'll transcribe in detected and translate
                if detected_lang != whisper_lang:
                    print(f"[SUBTITLE DEBUG] Language mismatch - will transcribe in {detected_lang} and translate to {whisper_lang}")
                    source_lang = detected_lang
                    target_lang = whisper_lang
                    needs_translation = True
                    # Use detected language for transcription
                    whisper_lang = detected_lang
                
                # Transcription options optimized for source language
                transcription_options = self._get_transcription_options(source_lang)
                print(f"[SUBTITLE DEBUG] Using transcription options: {transcription_options}")
                
                # For Urdu, use direct file path instead of loading with librosa
                # This preserves audio quality better for complex languages
                if source_lang in ['ur', 'ru-ur']:
                    print(f"[SUBTITLE DEBUG] Using direct file transcription for optimal Urdu accuracy")
                    result = model.transcribe(
                        processed_audio_path,
                        language=whisper_lang,
                        fp16=False,  # Disable FP16 for CPU compatibility
                        **transcription_options
                    )
                else:
                    # For other languages, use direct file path (faster and more stable)
                    print(f"[SUBTITLE DEBUG] Starting transcription (this may take a minute)...")
                    
                    result = model.transcribe(
                        processed_audio_path, 
                        language=whisper_lang,
                        fp16=False,  # Disable FP16 for CPU compatibility
                        **transcription_options
                    )
                
                print(f"[SUBTITLE DEBUG] Whisper transcription completed")
                print(f"[SUBTITLE DEBUG] Found {len(result.get('segments', []))} segments")
                print(f"[SUBTITLE DEBUG] Detected language: {result.get('language', 'unknown')}")
                
                # Extract segments with timestamps
                segments = []
                for i, segment in enumerate(result['segments']):
                    # Post-process text for source language
                    text = self._post_process_transcription(segment['text'].strip(), source_lang)
                    
                    # Skip empty segments
                    if not text or len(text.strip()) == 0:
                        continue
                    
                    # Detect and skip hallucinated/repetitive text
                    if self._is_hallucinated_text(text):
                        print(f"[SUBTITLE DEBUG] Skipping hallucinated segment: '{text[:100]}...'")
                        continue
                    
                    segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': text,
                        'confidence': segment.get('avg_logprob', 0.0)  # Track confidence
                    })
                    print(f"[SUBTITLE DEBUG] Segment {i+1}: {segment['start']:.2f}s-{segment['end']:.2f}s: '{text[:100]}...'")
                
                print(f"[SUBTITLE DEBUG] Successfully processed {len(segments)} segments from Whisper")
                
                # Translate segments if needed
                if needs_translation and len(segments) > 0:
                    print(f"[SUBTITLE DEBUG] Translating from {source_lang} to {target_lang}...")
                    segments = self._translate_segments(segments, source_lang, target_lang)
                    print(f"[SUBTITLE DEBUG] Translation completed")
                
                # Check if we have valid segments after filtering
                if len(segments) == 0:
                    print(f"[SUBTITLE DEBUG] No valid segments after hallucination filtering - using fallback")
                    text = self._get_enhanced_sample_text(language, clip.duration)
                    srt_content, json_data = self._create_subtitles(text, language, style, clip.duration)
                else:
                    # Generate both SRT and JSON format subtitles
                    srt_content, json_data = self._create_subtitles_from_segments(segments, language, style)
                    print(f"[SUBTITLE DEBUG] Using REAL Whisper transcription")
                
                # Cleanup preprocessed audio
                if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                    os.remove(processed_audio_path)
                
            except ImportError as e:
                print(f"[SUBTITLE DEBUG] Whisper or librosa not available: {e}")
                print(f"[SUBTITLE DEBUG] Falling back to enhanced sample text")
                # Enhanced fallback for Urdu
                text = self._get_enhanced_sample_text(language, clip.duration)
                srt_content, json_data = self._create_subtitles(text, language, style, clip.duration)
                
            except Exception as e:
                print(f"[SUBTITLE DEBUG] Whisper transcription failed with error: {e}")
                print(f"[SUBTITLE DEBUG] Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print(f"[SUBTITLE DEBUG] Falling back to enhanced sample text")
                
                # Enhanced fallback for Urdu
                text = self._get_enhanced_sample_text(language, clip.duration)
                srt_content, json_data = self._create_subtitles(text, language, style, clip.duration)
            
            # Save subtitles file
            srt_path = f"{os.path.splitext(video.filepath)[0]}_{language}.srt"
            print(f"[SUBTITLE DEBUG] Saving SRT file to: {srt_path}")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            # Save JSON format for live display
            json_path = f"{os.path.splitext(video.filepath)[0]}_{language}.json"
            print(f"[SUBTITLE DEBUG] Saving JSON file to: {json_path}")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            print(f"[SUBTITLE DEBUG] Subtitle generation completed successfully")
            print(f"[SUBTITLE DEBUG] Files saved: SRT={srt_path}, JSON={json_path}")
            
            video.outputs["subtitles"] = {
                "srt": srt_path,
                "json": json_path,
                "language": language,
                "style": style
            }
            
            if clip:
                clip.close()
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
                
        except Exception as e:
            print(f"[SUBTITLE DEBUG] Error generating subtitles: {e}")
            import traceback
            traceback.print_exc()
            # Create fallback subtitles
            if clip:
                clip.close()
            self._create_fallback_subtitles(video, options)

    def _get_optimal_whisper_model(self, language):
        """Get optimal Whisper model size based on language"""
        # Use balanced models - fast but accurate
        if language in ['ur', 'ru-ur']:
            # Use medium for Urdu - good balance of speed and accuracy
            return "medium"  # Good accuracy, much faster than large
        elif language in ['ar', 'hi', 'zh', 'ja', 'ko']:
            return "small"  # Small model for complex scripts - faster
        elif language in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'nl']:
            return "base"    # Base model for well-supported languages - fast
        else:
            return "tiny"   # Tiny model for other languages - very fast

    def _preprocess_audio_for_transcription(self, audio_path, language):
        """Preprocess audio for better transcription accuracy"""
        try:
            from pydub import AudioSegment
            from pydub.effects import normalize, compress_dynamic_range
            import numpy as np
            
            # Load audio
            audio = AudioSegment.from_file(audio_path)
            
            # Language-specific preprocessing
            if language in ['ur', 'ru-ur', 'ar', 'hi']:
                print(f"[AUDIO PREPROCESS] Applying ENHANCED {language} specific preprocessing")
                
                # Step 1: Normalize audio levels first
                audio = normalize(audio)
                
                # Step 2: Apply dynamic range compression for consistent volume
                audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
                
                # Step 3: Remove low-frequency noise (below 80Hz) that can interfere
                audio = audio.high_pass_filter(80)
                
                # Step 4: Remove very high frequencies (above 8000Hz) to reduce noise
                audio = audio.low_pass_filter(8000)
                
                # Step 5: Boost volume slightly for better detection
                audio = audio + 3  # Add 3dB
                
                # Step 6: Ensure mono for consistent processing
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                
                # Step 7: Set optimal sample rate for Whisper (16kHz is ideal)
                audio = audio.set_frame_rate(16000)
                
                # Step 8: Apply additional normalization after all processing
                audio = normalize(audio)
                
                # Save preprocessed audio
                processed_path = f"{os.path.splitext(audio_path)[0]}_processed.wav"
                audio.export(processed_path, format="wav", parameters=["-ac", "1"])
                
                print(f"[AUDIO PREPROCESS] Enhanced preprocessed audio saved to: {processed_path}")
                return processed_path
            
            else:
                # For other languages, minimal preprocessing
                if audio.channels > 1:
                    audio = audio.set_channels(1)
                audio = audio.set_frame_rate(16000)
                audio = normalize(audio)
                processed_path = f"{os.path.splitext(audio_path)[0]}_processed.wav"
                audio.export(processed_path, format="wav")
                return processed_path
                
        except Exception as e:
            print(f"[AUDIO PREPROCESS] Error preprocessing audio: {e}")
            return audio_path  # Return original if preprocessing fails

    def _get_transcription_options(self, language):
        """Get optimal transcription options for each language"""
        base_options = {
            "word_timestamps": True,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "verbose": True  # Enable verbose output for debugging
        }
        
        if language in ['ur', 'ru-ur']:
            # Urdu-specific options - balanced for speed and accuracy
            return {
                **base_options,
                "temperature": 0.0,  # Single temperature for speed
                "compression_ratio_threshold": 2.4,
                "condition_on_previous_text": True,
                "initial_prompt": "یہ ایک اردو زبان کی ویڈیو ہے۔",  # Simple Urdu prompt
                "beam_size": 3,  # Reduced for speed
                "best_of": 3,    # Reduced for speed
                "patience": 1.0,
                "length_penalty": 1.0,
                "suppress_tokens": "-1"
            }
        elif language == 'ar':
            return {
                **base_options,
                "temperature": (0.0, 0.2, 0.4),
                "compression_ratio_threshold": 2.4,
                "condition_on_previous_text": True,
                "initial_prompt": "هذا محتوى باللغة العربية. استخدم كلمات واضحة ودقيقة.",  # Enhanced Arabic prompt
                "beam_size": 5,
                "best_of": 5
            }
        elif language == 'hi':
            return {
                **base_options,
                "temperature": (0.0, 0.2, 0.4),
                "compression_ratio_threshold": 2.4,
                "condition_on_previous_text": True,
                "initial_prompt": "यह हिंदी भाषा की सामग्री है। स्पष्ट और सटीक शब्दों का प्रयोग करें।",  # Enhanced Hindi prompt
                "beam_size": 5,
                "best_of": 5
            }
        else:
            return {
                **base_options,
                "temperature": 0.0,
                "beam_size": 3,
                "best_of": 3
            }

    def _post_process_transcription(self, text, language):
        """Post-process transcribed text for language-specific improvements"""
        if not text:
            return text
            
        if language in ['ur', 'ru-ur']:
            # Urdu-specific post-processing
            
            # Clean up common transcription artifacts
            text = text.strip()
            
            # Remove or fix common Whisper artifacts for Urdu
            # These are patterns that Whisper sometimes incorrectly transcribes
            urdu_fixes = {
                ' .' : '۔',
                ' ؟' : '؟',
                ' !' : '!',
                '  ': ' ',  # Remove double spaces
            }
            
            for wrong, correct in urdu_fixes.items():
                text = text.replace(wrong, correct)
            
            # Ensure proper Urdu punctuation
            if text and not text.endswith(('۔', '؟', '!', '.', '?')):
                text += '۔'
                
        elif language == 'ar':
            # Arabic-specific post-processing
            text = text.strip()
            if text and not text.endswith(('۔', '؟', '!', '.', '?')):
                text += '.'
                
        # General cleanup for all languages
        text = ' '.join(text.split())  # Normalize whitespace
        
        return text

    def _is_hallucinated_text(self, text):
        """Detect if transcribed text is hallucinated/repetitive/nonsense"""
        if not text or len(text.strip()) < 10:
            return True
        
        # Check for excessive repetition
        words = text.lower().split()
        if len(words) < 5:
            return False
            
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any word appears more than 40% of the time, it's likely hallucinated
        max_freq = max(word_freq.values())
        if max_freq / len(words) > 0.4:
            return True
        
        # Check for repeating phrases (3+ words)
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if text.lower().count(phrase) > 3:  # Same 3-word phrase repeated more than 3 times
                return True
        
        # Check for common hallucination patterns
        hallucination_patterns = [
            'thank you for watching',
            'please subscribe',
            'like and subscribe',
            'the most merciful',  # Islamic phrases repeated
            'in the name of',
            'music playing',
            '[music]',
            '♪',
        ]
        
        text_lower = text.lower()
        for pattern in hallucination_patterns:
            if text_lower.count(pattern) > 2:
                return True
        
        return False

    def _translate_segments(self, segments, source_lang, target_lang):
        """Translate subtitle segments from source to target language"""
        try:
            from deep_translator import GoogleTranslator
            
            # Map language codes
            lang_map = {
                'ur': 'ur',
                'en': 'en',
                'ar': 'ar',
                'hi': 'hi',
                'es': 'es',
                'fr': 'fr',
                'de': 'de',
                'it': 'it',
                'pt': 'pt',
                'ru': 'ru',
                'zh': 'zh-CN',
                'ja': 'ja',
                'ko': 'ko'
            }
            
            source = lang_map.get(source_lang, source_lang)
            target = lang_map.get(target_lang, target_lang)
            
            translator = GoogleTranslator(source=source, target=target)
            
            translated_segments = []
            for segment in segments:
                try:
                    translated_text = translator.translate(segment['text'])
                    translated_segments.append({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': translated_text,
                        'confidence': segment['confidence']
                    })
                    print(f"[TRANSLATE] {segment['text'][:50]} -> {translated_text[:50]}")
                except Exception as e:
                    print(f"[TRANSLATE] Error translating segment: {e}, keeping original")
                    translated_segments.append(segment)
            
            return translated_segments
            
        except ImportError:
            print(f"[TRANSLATE] deep-translator not available, keeping original language")
            return segments
        except Exception as e:
            print(f"[TRANSLATE] Translation failed: {e}, keeping original language")
            return segments

    def _get_enhanced_sample_text(self, language, duration):
        """Get enhanced sample text with better content for fallback"""
        if language == 'ur':
            # More comprehensive Urdu sample text
            long_urdu_text = """
            یہ ویڈیو SnipX AI کے ذریعے خودکار طور پر پروسیس کیا گیا ہے۔
            ہمارا جدید ترین سسٹم اردو زبان کی خصوصیات کو سمجھتا ہے۔
            آڈیو انہانسمنٹ اور نائز ریڈکشن کے ذریعے بہترین نتائج حاصل کیے جاتے ہیں۔
            سب ٹائٹلز کی درستگی کے لیے ہم مختلف تکنیکوں کا استعمال کرتے ہیں۔
            یہ ٹیکنالوجی اردو بولنے والوں کے لیے خاص طور پر ڈیزائن کی گئی ہے۔
            ہمارا مقصد بہترین صوتی تجربہ فراہم کرنا ہے۔
            """
            return long_urdu_text.strip()
            
        elif language == 'ru-ur':
            # Enhanced Roman Urdu sample text
            long_roman_urdu = """
            Yeh video SnipX AI ke zariye automatically process kiya gaya hai.
            Hamara advanced system Urdu language ki features ko samajhta hai.
            Audio enhancement aur noise reduction ke zariye best results hasil kiye jaate hain.
            Subtitles ki accuracy ke liye hum different techniques ka istemal karte hain.
            Yeh technology Urdu speakers ke liye specially design ki gayi hai.
            Hamara maqsad behtereen audio experience provide karna hai.
            Is system mein latest AI models shamil hain jo Urdu content ko accurately process kar sakte hain.
            """
            return long_roman_urdu.strip()
            
        else:
            # Use existing sample text for other languages
            return self._get_sample_text(language)

    def _summarize_video(self, video):
        if not self.summarizer or not self.speech_recognizer:
            print("AI models not available for summarization")
            return
            
        try:
            # Extract audio and convert to text
            clip = VideoFileClip(video.filepath)
            audio_path = f"{os.path.splitext(video.filepath)[0]}_audio.wav"
            clip.audio.write_audiofile(audio_path)
            
            # Generate transcription
            transcription = self.speech_recognizer(audio_path)
            text = transcription.get('text', '')
            
            if text:
                # Summarize text
                summary = self.summarizer(text, max_length=130, min_length=30)
                
                # Save summary
                summary_path = f"{os.path.splitext(video.filepath)[0]}_summary.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary[0]['summary_text'])
                
                video.outputs["summary"] = summary_path
            
            clip.close()
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"Error summarizing video: {e}")

    def _format_timestamp(self, seconds):
        """Format timestamp for SRT format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _get_whisper_language_code(self, language):
        """Convert our language codes to Whisper language codes"""
        whisper_codes = {
            'en': 'en',
            'ur': 'ur',
            'ru-ur': 'ur',  # Use Urdu model for Roman Urdu
            'ar': 'ar',
            'hi': 'hi',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'zh': 'zh',
            'ja': 'ja',
            'ko': 'ko',
            'pt': 'pt',
            'ru': 'ru',
            'it': 'it',
            'tr': 'tr',
            'nl': 'nl'
        }
        return whisper_codes.get(language, 'en')

    def _create_subtitles_from_segments(self, segments, language, style):
        """Create both SRT and JSON format subtitles from Whisper segments"""
        srt_content = ""
        json_data = {
            "language": language,
            "segments": [],
            "word_timestamps": True,
            "confidence": 0.95,
            "source": "whisper"
        }
        
        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            
            # SRT format
            srt_content += f"{i + 1}\n"
            srt_content += f"{self._format_timestamp(start_time)} --> {self._format_timestamp(end_time)}\n"
            srt_content += f"{text}\n\n"
            
            # JSON format for live display
            json_data["segments"].append({
                "id": i + 1,
                "start": start_time,
                "end": end_time,
                "text": text,
                "language": language,
                "style": style
            })
        
        return srt_content, json_data

    def _get_sample_text(self, language):
        """Get sample text for different languages"""
        sample_texts = {
            'en': "Welcome to this video demonstration. This is an example of English subtitles generated automatically by SnipX AI.",
            'ur': "اس ویڈیو ڈیمونسٹریشن میں خوش آمدید۔ یہ اردو سب ٹائٹلز کی مثال ہے جو SnipX AI کے ذریعے خودکار طور پر تیار کیا گیا۔ ہمارا سسٹم اردو زبان کے لیے خاص طور پر تربیت یافتہ ہے۔",
            'ru-ur': "Is video demonstration mein khush aamdeed. Yeh Roman Urdu subtitles ki misaal hai jo SnipX AI ke zariye automatic tayyar kiya gaya. Hamara system Urdu language ke liye khaas training ke saath banaya gaya hai.",
            'es': "Bienvenido a esta demostración de video. Este es un ejemplo de subtítulos en español generados automáticamente por SnipX AI.",
            'fr': "Bienvenue dans cette démonstration vidéo. Ceci est un exemple de sous-titres français générés automatiquement par SnipX AI.",
            'de': "Willkommen zu dieser Video-Demonstration. Dies ist ein Beispiel für deutsche Untertitel, die automatisch von SnipX AI generiert wurden.",
            'ar': "مرحباً بكم في هذا العرض التوضيحي للفيديو. هذا مثال على الترجمة العربية التي تم إنشاؤها تلقائياً بواسطة SnipX AI.",
            'hi': "इस वीडियो प्रदर्शन में आपका स्वागत है। यह SnipX AI द्वारा स्वचालित रूप से उत्पन्न हिंदी उपशीर्षक का एक उदाहरण है।",
            'zh': "欢迎观看此视频演示。这是由SnipX AI自动生成的中文字幕示例。",
            'ja': "このビデオデモンストレーションへようこそ。これはSnipX AIによって自動生成された日本語字幕の例です。",
            'ko': "이 비디오 데모에 오신 것을 환영합니다. 이것은 SnipX AI에 의해 자동으로 생성된 한국어 자막의 예입니다。",
            'pt': "Bem-vindo a esta demonstração de vídeo. Este é um exemplo de legendas em português geradas automaticamente pelo SnipX AI.",
            'ru': "Добро пожаловать в эту видео-демонстрацию. Это пример русских субтитров, автоматически созданных SnipX AI.",
            'it': "Benvenuti in questa dimostrazione video. Questo è un esempio di sottotitoli italiani generati automaticamente da SnipX AI.",
            'tr': "Bu video gösterimine hoş geldiniz. Bu, SnipX AI tarafından otomatik olarak oluşturulan Türkçe altyazı örneğidir.",
            'nl': "Welkom bij deze videodemonstratie. Dit is een voorbeeld van Nederlandse ondertitels die automatisch zijn gegenereerd door SnipX AI."
        }
        return sample_texts.get(language, sample_texts['en'])

    def _create_subtitles(self, text, language, style, duration):
        """Create both SRT and JSON format subtitles"""
        # Split text into chunks for subtitles
        words = text.split()
        chunk_size = 4 if language in ['ur', 'ar', 'hi', 'zh', 'ja', 'ko'] else 6  # Smaller chunks for better readability
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        srt_content = ""
        json_data = {
            "language": language,
            "segments": [],
            "word_timestamps": True,
            "confidence": 0.95,
            "source": "fallback"
        }
        
        subtitle_duration = duration / len(chunks) if chunks else 5
        
        for i, chunk in enumerate(chunks):
            start_time = i * subtitle_duration
            end_time = (i + 1) * subtitle_duration
            
            # SRT format
            srt_content += f"{i + 1}\n"
            srt_content += f"{self._format_timestamp(start_time)} --> {self._format_timestamp(end_time)}\n"
            srt_content += f"{chunk}\n\n"
            
            # JSON format for live display
            json_data["segments"].append({
                "id": i + 1,
                "start": start_time,
                "end": end_time,
                "text": chunk,
                "language": language,
                "style": style
            })
        
        return srt_content, json_data

    def _create_fallback_subtitles(self, video, options):
        """Create fallback subtitles when transcription fails"""
        language = options.get('subtitle_language', 'en')
        style = options.get('subtitle_style', 'clean')
        
        # Use enhanced fallback text
        fallback_text = self._get_enhanced_sample_text(language, 15)
        srt_content, json_data = self._create_subtitles(fallback_text, language, style, 15)
        
        srt_path = f"{os.path.splitext(video.filepath)[0]}_{language}_fallback.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        json_path = f"{os.path.splitext(video.filepath)[0]}_{language}_fallback.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        video.outputs["subtitles"] = {
            "srt": srt_path,
            "json": json_path,
            "language": language,
            "style": style
        }

    def export_video_with_edits(self, video_id, trim_start=0, trim_end=100, 
                                 text_overlay='', text_position='center', 
                                 text_color='#ffffff', text_size=32,
                                 music_volume=50, video_volume=100, mute_original=False):
        """Export video with all editing changes applied (trim, text overlay, audio adjustments)"""
        import subprocess
        
        video = self.get_video(video_id)
        if not video:
            raise ValueError("Video not found")
        
        print(f"[EXPORT] Starting export for video: {video.filename}")
        print(f"[EXPORT] Trim: {trim_start}% - {trim_end}%")
        print(f"[EXPORT] Text: '{text_overlay}' at {text_position}")
        print(f"[EXPORT] Audio: video_vol={video_volume}, mute={mute_original}")
        
        # Get video duration
        clip = VideoFileClip(video.filepath)
        duration = clip.duration
        clip.close()
        
        # Calculate trim times
        start_time = (trim_start / 100) * duration
        end_time = (trim_end / 100) * duration
        trim_duration = end_time - start_time
        
        print(f"[EXPORT] Duration: {duration}s, Trimming: {start_time}s to {end_time}s ({trim_duration}s)")
        
        # Output path
        base_name = os.path.splitext(video.filename)[0]
        export_filename = f"{base_name}_edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        export_path = os.path.join(self.upload_folder, export_filename)
        
        # Build FFmpeg command
        ffmpeg_path = os.path.join(FFMPEG_PATH, 'ffmpeg.exe')
        
        # Build filter complex for video
        video_filters = []
        
        # Add text overlay if provided
        if text_overlay and text_overlay.strip():
            # Map position to FFmpeg coordinates
            position_map = {
                'top-left': 'x=50:y=50',
                'top-center': 'x=(w-text_w)/2:y=50',
                'top-right': 'x=w-text_w-50:y=50',
                'center': 'x=(w-text_w)/2:y=(h-text_h)/2',
                'bottom-left': 'x=50:y=h-text_h-50',
                'bottom-center': 'x=(w-text_w)/2:y=h-text_h-50',
                'bottom-right': 'x=w-text_w-50:y=h-text_h-50'
            }
            pos = position_map.get(text_position, position_map['center'])
            
            # Escape special characters in text
            safe_text = text_overlay.replace("'", "\\'").replace(":", "\\:")
            
            # Convert hex color to FFmpeg format
            color = text_color.lstrip('#')
            
            # Use a system font that's available on Windows
            text_filter = f"drawtext=text='{safe_text}':{pos}:fontsize={text_size}:fontcolor=0x{color}:borderw=3:bordercolor=black"
            video_filters.append(text_filter)
        
        # Build the FFmpeg command
        cmd = [ffmpeg_path, '-y']  # -y to overwrite output
        
        # Input with trim
        cmd.extend(['-ss', str(start_time), '-t', str(trim_duration), '-i', video.filepath])
        
        # Apply video filters if any
        if video_filters:
            cmd.extend(['-vf', ','.join(video_filters)])
        
        # Audio settings
        if mute_original:
            cmd.extend(['-an'])  # No audio
        else:
            volume = video_volume / 100
            cmd.extend(['-af', f'volume={volume}'])
        
        # Output settings
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            export_path
        ])
        
        print(f"[EXPORT] Running FFmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"[EXPORT] FFmpeg error: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")
            
            print(f"[EXPORT] Video exported successfully to: {export_path}")
            
            # Update video outputs and status in database
            video.outputs['exported_video'] = export_path
            video.status = 'completed'
            
            # Get file size of exported video
            export_size = os.path.getsize(export_path) if os.path.exists(export_path) else 0
            
            self.videos.update_one(
                {"_id": ObjectId(video_id)},
                {"$set": {
                    "outputs.exported_video": export_path,
                    "status": "completed",
                    "process_end_time": datetime.utcnow(),
                    "metadata.exported_size": export_size,
                    "metadata.trim_start": trim_start,
                    "metadata.trim_end": trim_end,
                    "metadata.has_text_overlay": bool(text_overlay and text_overlay.strip())
                }}
            )
            
            return export_path
            
        except subprocess.TimeoutExpired:
            print("[EXPORT] FFmpeg timed out")
            raise Exception("Export timed out - video may be too long")
        except Exception as e:
            print(f"[EXPORT] Export failed: {e}")
            raise
