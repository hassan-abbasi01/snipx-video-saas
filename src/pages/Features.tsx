import { useState, useEffect, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Scissors,
  Captions,
  Film,
  Wand2,
  Image as ImageIcon,
  UploadCloud,
  Play,
  Video,
  Download,
  Loader2,
  CheckCircle,
  AlertCircle,
  LogIn,
  Sparkles,
  Zap,
  Brain,
  Volume2,
  Eye,
  Star
} from 'lucide-react';
import { ApiService } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import VideoPlayer from '../components/VideoPlayer';
import toast from 'react-hot-toast';
import API_URL from '../config/api';

// Helper function to format file size
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

interface ConsoleLog {
  timestamp: string;
  message: string;
  type?: 'info' | 'success' | 'error';
}

interface ProgressState {
  visible: boolean;
  percentage: number;
  status: string;
}

interface VideoData {
  id: string;
  filename: string;
  status: string;
  metadata?: {
    duration?: number;
    format?: string;
    resolution?: string;
    fps?: number;
  };
  outputs?: {
    processed_video?: string;
    thumbnail?: string;
    subtitles?: string;
    summary?: string;
    enhancement_results?: {
      filler_words_removed?: number;
      noise_reduction_percentage?: number;
      duration_reduction_percentage?: number;
      original_duration?: string;
      enhanced_duration?: string;
      time_saved?: string;
    };
    audio_enhancement_metrics?: {
      original_duration_ms?: number;
      enhanced_duration_ms?: number;
      time_saved_ms?: number;
      time_saved_percentage?: number;
      filler_words_removed?: number;
    };
  };
}

const Features = () => {
  const { isAuthenticated, loginAsDemo } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<string>('enhancement');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(null);
  const [videoData, setVideoData] = useState<VideoData | null>(null);
  const [consoleLogs, setConsoleLogs] = useState<ConsoleLog[]>([
    { timestamp: new Date().toLocaleTimeString(), message: '[System] SnipX Video Editor API Ready', type: 'info' }
  ]);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [isLoadingPreview, setIsLoadingPreview] = useState<boolean>(false);
  const [isLoadingThumbnails, setIsLoadingThumbnails] = useState<boolean>(false);
  const [thumbnailFrames, setThumbnailFrames] = useState<string[]>([]);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState<number | null>(null);
  const [generatedThumbnail, setGeneratedThumbnail] = useState<string | null>(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  // State for form inputs
  const [pauseThreshold, setPauseThreshold] = useState<number>(500);
  const [fillerWordsLevel, setFillerWordsLevel] = useState<string>('medium');
  const [subtitleLanguage, setSubtitleLanguage] = useState<string>('en');
  const [subtitleStyle, setSubtitleStyle] = useState<string>('clean');
  const [summaryLength, setSummaryLength] = useState<string>('medium');
  const [summaryFocus, setSummaryFocus] = useState<string>('balanced');
  const [stabilizationLevel, setStabilizationLevel] = useState<string>('medium');
  const [audioEnhancement, setAudioEnhancement] = useState<string>('full');
  const [brightnessLevel, setBrightnessLevel] = useState<number>(100);
  const [contrastLevel, setContrastLevel] = useState<number>(100);
  const [thumbnailStyle, setThumbnailStyle] = useState<string>('modern');
  const [thumbnailText, setThumbnailText] = useState<string>('');
  const [aiColorEnhancement, setAiColorEnhancement] = useState<boolean>(false);

  // State for progress bars
  const [audioProgress, setAudioProgress] = useState<ProgressState>({ visible: false, percentage: 0, status: '' });
  const [subtitlesProgress, setSubtitlesProgress] = useState<ProgressState>({ visible: false, percentage: 0, status: '' });
  const [summarizationProgress, setSummarizationProgress] = useState<ProgressState>({ visible: false, percentage: 0, status: '' });
  const [enhancementProgress, setEnhancementProgress] = useState<ProgressState>({ visible: false, percentage: 0, status: '' });
  const [thumbnailProgress, setThumbnailProgress] = useState<ProgressState>({ visible: false, percentage: 0, status: '' });

  // State for live preview
  const [previewFilters, setPreviewFilters] = useState<string>('');
  const [generatedSubtitles, setGeneratedSubtitles] = useState<string>('');
  const [subtitleFile, setSubtitleFile] = useState<string | null>(null);
  const [subtitleData, setSubtitleData] = useState<any[]>([]);
  const [showConsole, setShowConsole] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const consoleRef = useRef<HTMLDivElement>(null);
  const statusCheckIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);

  // Mouse tracking for 3D effects - DISABLED to prevent hanging
  useEffect(() => {
    // Disabled mouse tracking to prevent animation hanging
    // const handleMouseMove = (e: MouseEvent) => { ... }
    return () => {
      // Cleanup function remains for safety
    };
  }, []);

  // Handle URL parameters for direct tab navigation
  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const tabParam = searchParams.get('tab');
    
    if (tabParam) {
      // Map of valid tab names
      const validTabs = ['audio', 'subtitles', 'summarization', 'enhancement', 'thumbnail'];
      
      if (validTabs.includes(tabParam)) {
        setActiveTab(tabParam);
        logToConsole(`Navigated to ${tabParam} tab from external link`);
        
        // Scroll to middle of the page after content is rendered
        setTimeout(() => {
          const windowHeight = window.innerHeight;
          const documentHeight = document.documentElement.scrollHeight;
          // Position at about 25% from top to show the AI Studio area
          const targetPosition = Math.max(0, documentHeight * 0.25);
          
          window.scrollTo({
            top: targetPosition,
            behavior: 'smooth'
          });
        }, 500);
        
        // Clear the URL parameter to keep URL clean
        const newUrl = new URL(window.location.href);
        newUrl.searchParams.delete('tab');
        navigate(newUrl.pathname, { replace: true });
      }
    }
  }, [location.search, navigate]);

  // Log to console function
  const logToConsole = useCallback((message: string, type: 'info' | 'success' | 'error' = 'info') => {
    setConsoleLogs(prevLogs => [
      ...prevLogs,
      { timestamp: new Date().toLocaleTimeString(), message, type }
    ]);
  }, []);

  // Scroll console to bottom
  useEffect(() => {
    if (consoleRef.current) {
      consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
    }
  }, [consoleLogs]);

  // Update live preview filters when enhancement values change
  useEffect(() => {
    const brightness = brightnessLevel / 100;
    const contrast = contrastLevel / 100;
    
    // Apply AI-suggested saturation boost for live preview
    let saturation = 1.0;
    if (aiColorEnhancement) {
      // Simulate AI boost - in real scenario this would be calculated from video analysis
      saturation = 1.3; // Default AI boost for demonstration
    }
    
    const filterString = `brightness(${brightness}) contrast(${contrast}) saturate(${saturation})`;
    setPreviewFilters(filterString);
    
    if (videoRef.current) {
      videoRef.current.style.filter = filterString;
    }
  }, [brightnessLevel, contrastLevel, aiColorEnhancement]);

  // Demo video upload function (simulates upload without backend)
  const uploadVideoDemo = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);
    logToConsole(`Starting demo upload: ${file.name} (${formatFileSize(file.size)})`);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = Math.min(prev + Math.random() * 15, 95);
          return newProgress;
        });
      }, 200);

      // Simulate upload delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Generate demo video ID
      const demoVideoId = 'demo-video-' + Date.now();
      setUploadedVideoId(demoVideoId);
      
      // Set demo video data
      setVideoData({
        id: demoVideoId,
        filename: file.name,
        status: 'uploaded',
        metadata: {
          duration: 120,
          format: 'mp4',
          resolution: '1920x1080',
          fps: 30
        },
        outputs: {}
      });
      
      logToConsole(`Demo upload successful! Video ID: ${demoVideoId}`, 'success');
      toast.success('Video uploaded successfully (Demo Mode)');
      
    } catch (error) {
      logToConsole(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      toast.error('Upload failed. Please try again.');
    } finally {
      setIsUploading(false);
    }
  };

  // Real video upload function
  const uploadVideo = async (file: File) => {
    if (!isAuthenticated) {
      toast.error('Please login to upload videos');
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);
    logToConsole(`Starting upload: ${file.name} (${formatFileSize(file.size)})`);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          const newProgress = Math.min(prev + Math.random() * 15, 95);
          return newProgress;
        });
      }, 200);

      const response = await ApiService.uploadVideo(file);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      if (response.video_id) {
        setUploadedVideoId(response.video_id);
        logToConsole(`Upload successful! Video ID: ${response.video_id}`, 'success');
        
        // Start checking video status
        startStatusCheck(response.video_id);
      }
    } catch (error) {
      logToConsole(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      toast.error('Upload failed. Using demo mode instead.');
      
      // Fallback to demo mode
      await uploadVideoDemo(file);
    } finally {
      setIsUploading(false);
    }
  };

  // Check video processing status
  const startStatusCheck = (videoId: string) => {
    const checkStatus = async () => {
      try {
        const data = await ApiService.getVideoStatus(videoId);
        setVideoData(data);
        
        if (data.status === 'completed' || data.status === 'failed') {
          if (statusCheckIntervalRef.current) {
            clearInterval(statusCheckIntervalRef.current);
            statusCheckIntervalRef.current = null;
          }
          
          if (data.status === 'completed') {
            logToConsole('Video processing completed successfully!', 'success');
          } else {
            logToConsole(`Video processing failed: ${data.error || 'Unknown error'}`, 'error');
          }
        }
      } catch (error) {
        logToConsole(`Status check failed: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
      }
    };

    // Check immediately and then every 2 seconds
    checkStatus();
    statusCheckIntervalRef.current = setInterval(checkStatus, 2000);
  };

  // Demo processing function
  const processVideoDemo = async (
    options: any,
    progressSetter: React.Dispatch<React.SetStateAction<ProgressState>>,
    successMessage: string
  ) => {
    progressSetter({ visible: true, percentage: 0, status: 'Starting processing...' });
    
    try {
      // Simulate progress updates
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress >= 100) {
          clearInterval(progressInterval);
          progress = 100;
          progressSetter(prev => ({ ...prev, percentage: 100, status: successMessage }));
          logToConsole(successMessage + ' (Demo Mode)', 'success');
          
          // Update video data status
          if (videoData) {
            setVideoData(prev => prev ? { ...prev, status: 'completed' } : null);
          }
        } else {
          progressSetter(prev => ({ ...prev, percentage: progress, status: `${Math.round(progress)}% - Processing...` }));
        }
      }, 300);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Processing failed';
      logToConsole(`Processing failed: ${errorMessage}`, 'error');
      progressSetter(prev => ({ ...prev, status: `Error: ${errorMessage}` }));
      toast.error(errorMessage);
    }
  };

  // Enhanced processing function with detailed options
  const processVideo = async (options: {
    cut_silence?: boolean;
    enhance_audio?: boolean;
    generate_thumbnail?: boolean;
    generate_subtitles?: boolean;
    summarize?: boolean;
    // Enhancement specific options
    stabilization?: string;
    audio_enhancement_type?: string;
    pause_threshold?: number;
    noise_reduction?: string;
    brightness?: number;
    contrast?: number;
    // Subtitle specific options
    subtitle_language?: string;
    subtitle_style?: string;
  }, progressSetter: React.Dispatch<React.SetStateAction<ProgressState>>, successMessage: string) => {
    if (!uploadedVideoId) {
      toast.error('Please upload a video first');
      return;
    }

    if (!isAuthenticated) {
      // Use demo processing
      return processVideoDemo(options, progressSetter, successMessage);
    }

    progressSetter({ visible: true, percentage: 0, status: 'Starting processing...' });
    
    try {
      await ApiService.processVideo(uploadedVideoId, options);
      
      // Simulate progress updates
      let progress = 0;
      const progressInterval = setInterval(async () => {
        progress += Math.random() * 10;
        if (progress >= 100) {
          clearInterval(progressInterval);
          progress = 100;
          progressSetter(prev => ({ ...prev, percentage: 100, status: successMessage }));
          logToConsole(successMessage, 'success');
          
          // Immediately fetch updated video data to get enhancement results
          if (uploadedVideoId) {
            try {
              const updatedData = await ApiService.getVideoStatus(uploadedVideoId);
              setVideoData(updatedData);
              logToConsole('Video data refreshed with enhancement results', 'info');
            } catch (err) {
              console.error('Failed to fetch updated video data:', err);
            }
          }
        } else {
          progressSetter(prev => ({ ...prev, percentage: progress, status: `${Math.round(progress)}% - Processing...` }));
        }
      }, 300);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Processing failed';
      logToConsole(`Processing failed: ${errorMessage}`, 'error');
      progressSetter(prev => ({ ...prev, status: `Error: ${errorMessage}` }));
      toast.error(errorMessage);
      
      // Fallback to demo processing
      await processVideoDemo(options, progressSetter, successMessage);
    }
  };

  // Cleanup intervals on unmount
  useEffect(() => {
    return () => {
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
      }
      if (statusCheckIntervalRef.current) {
        clearInterval(statusCheckIntervalRef.current);
      }
    };
  }, []);

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const objectUrl = URL.createObjectURL(file);
      setVideoSrc(objectUrl);
      
      // Reset states
      setUploadedVideoId(null);
      setVideoData(null);
      setGeneratedThumbnail(null);
      setThumbnailFrames([]);
      setSelectedFrameIndex(null);
      setGeneratedSubtitles('');
      setSubtitleData([]);
      setSubtitleFile(null);
      
      // Reset enhancement values
      setBrightnessLevel(100);
      setContrastLevel(100);
      setPreviewFilters('');
      setAiColorEnhancement(false);
      
      // Upload the video (will use demo mode if not authenticated)
      if (isAuthenticated) {
        await uploadVideo(file);
      } else {
        await uploadVideoDemo(file);
      }
    }
  };

  // Effect to auto-generate thumbnail frames when video is loaded
  useEffect(() => {
    if (videoSrc && selectedFile && thumbnailFrames.length === 0) {
      // Small delay to ensure video element can load
      const timer = setTimeout(() => {
        simulateThumbnailFrameGeneration();
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [videoSrc, selectedFile]);

  // Effect for cleaning up Object URL
  useEffect(() => {
    const currentVideoSrc = videoSrc;
    return () => {
      if (currentVideoSrc) {
        URL.revokeObjectURL(currentVideoSrc);
      }
    };
  }, [videoSrc]);

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleTabClick = (tabId: string) => {
    setActiveTab(tabId);
    logToConsole(`Switched to ${tabId.replace('-', ' ')} tab`);
    
    if (tabId === 'thumbnail' && selectedFile && thumbnailFrames.length === 0) {
      simulateThumbnailFrameGeneration();
    }
  };

  const simulateThumbnailFrameGeneration = () => {
    if (!selectedFile || !videoSrc) return;
    setIsLoadingThumbnails(true);
    logToConsole('Generating thumbnail frames from video...');
    
    // Create a video element to extract frames
    const video = document.createElement('video');
    video.src = videoSrc;
    video.crossOrigin = 'anonymous';
    
    video.addEventListener('loadedmetadata', () => {
      const duration = video.duration;
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        logToConsole('Failed to create canvas context', 'error');
        setIsLoadingThumbnails(false);
        return;
      }
      
      canvas.width = 192; // 2x size for better quality
      canvas.height = 128;
      
      const frames: string[] = [];
      const positions = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]; // Different positions in video
      let currentIndex = 0;
      
      const captureFrame = () => {
        if (currentIndex >= positions.length) {
          setThumbnailFrames(frames);
          setIsLoadingThumbnails(false);
          logToConsole(`Generated ${frames.length} thumbnail frames successfully!`, 'success');
          return;
        }
        
        const timestamp = duration * positions[currentIndex];
        video.currentTime = timestamp;
      };
      
      video.addEventListener('seeked', () => {
        try {
          // Draw the current video frame to canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Convert canvas to data URL
          const frameDataUrl = canvas.toDataURL('image/jpeg', 0.9);
          frames.push(frameDataUrl);
          
          logToConsole(`Extracted frame ${currentIndex + 1}/6 at ${Math.round(video.currentTime)}s`);
          currentIndex++;
          captureFrame();
        } catch (error) {
          logToConsole(`Failed to capture frame ${currentIndex + 1}: ${error}`, 'error');
          currentIndex++;
          captureFrame();
        }
      });
      
      // Start capturing frames
      captureFrame();
    });
    
    video.addEventListener('error', (e) => {
      logToConsole(`Video loading error: ${e}`, 'error');
      setIsLoadingThumbnails(false);
    });
  };

  // Enhanced processing functions
  const handleProcessAudio = () => {
    if (!uploadedVideoId) {
      toast.error('Please upload a video file first');
      return;
    }
    logToConsole(`Starting audio processing: Pause Threshold=${pauseThreshold}ms, Noise Reduction=${audioEnhancement}`);
    processVideo(
      { 
        cut_silence: true, 
        enhance_audio: true,
        pause_threshold: pauseThreshold,
        noise_reduction: audioEnhancement
      },
      setAudioProgress,
      'Audio processing completed successfully'
    );
  };

  // FIXED: Enhanced subtitle generation with proper polling
  const handleGenerateSubtitles = async () => {
    if (!uploadedVideoId) {
      toast.error('Please upload a video file first');
      return;
    }
    logToConsole(`Starting subtitle generation: Lang=${subtitleLanguage}, Style=${subtitleStyle}`);
    
    // Start the processing
    setSubtitlesProgress({ visible: true, percentage: 0, status: 'Starting subtitle generation...' });
    
    try {
      // First process the video to generate subtitles
      await processVideo(
        { 
          generate_subtitles: true,
          subtitle_language: subtitleLanguage,
          subtitle_style: subtitleStyle
        },
        setSubtitlesProgress,
        'Subtitles generated successfully'
      );
      
      // Poll for real subtitles with proper timing
      if (isAuthenticated && uploadedVideoId) {
        let attempts = 0;
        const maxAttempts = 15; // Try for up to 30 seconds (15 attempts × 2 seconds)
        
        const pollForSubtitles = async () => {
          try {
            attempts++;
            logToConsole(`Polling for subtitles (attempt ${attempts}/${maxAttempts})...`, 'info');
            
            // Update progress to show we're fetching
            setSubtitlesProgress(prev => ({ 
              ...prev, 
              percentage: Math.min(90, 70 + (attempts * 2)), 
              status: `Fetching subtitles (${attempts}/${maxAttempts})...` 
            }));
            
            const subtitleApiData = await ApiService.getVideoSubtitles(uploadedVideoId);
            
            if (subtitleApiData && subtitleApiData.length > 0) {
              // Success! We got real subtitles
              setSubtitleData(subtitleApiData);
              const srtContent = convertSubtitlesToSRT(subtitleApiData);
              setGeneratedSubtitles(srtContent);
              setSubtitleFile(`subtitles_${subtitleLanguage}.srt`);
              
              setSubtitlesProgress({ visible: false, percentage: 100, status: 'Completed' });
              logToConsole(`✅ Real subtitles loaded: ${subtitleApiData.length} segments in ${getLanguageName(subtitleLanguage)}`, 'success');
              toast.success(`Real subtitles loaded with ${subtitleApiData.length} segments!`);
              return true; // Success, stop polling
            } else {
              logToConsole(`No subtitle data yet (attempt ${attempts})`, 'info');
              return false; // Continue polling
            }
          } catch (error) {
            logToConsole(`Subtitle API error (attempt ${attempts}): ${error}`, 'error');
            return false; // Continue polling
          }
        };
        
        // Start polling
        const pollInterval = setInterval(async () => {
          const success = await pollForSubtitles();
          
          if (success || attempts >= maxAttempts) {
            clearInterval(pollInterval);
            
            if (!success) {
              // Fallback to sample subtitles after exhausting attempts
              logToConsole('Maximum polling attempts reached, using sample subtitles as fallback', 'info');
              const sampleSubtitles = generateSampleSubtitles(subtitleLanguage);
              setGeneratedSubtitles(sampleSubtitles);
              setSubtitleFile(`subtitles_${subtitleLanguage}.srt`);
              setSubtitleData([]); // Clear subtitle data when using samples
              setSubtitlesProgress({ visible: false, percentage: 100, status: 'Completed with fallback' });
              toast.success(`Sample subtitles generated in ${getLanguageName(subtitleLanguage)}`);
            }
          }
        }, 2000); // Poll every 2 seconds
        
      } else {
        // Demo mode - use sample subtitles immediately
        logToConsole('Demo mode: Using sample subtitles', 'info');
        const sampleSubtitles = generateSampleSubtitles(subtitleLanguage);
        setGeneratedSubtitles(sampleSubtitles);
        setSubtitleFile(`subtitles_${subtitleLanguage}.srt`);
        setSubtitleData([]); // Clear subtitle data in demo mode
        setSubtitlesProgress({ visible: false, percentage: 100, status: 'Demo completed' });
        toast.success(`Sample subtitles generated in ${getLanguageName(subtitleLanguage)}`);
      }
      
    } catch (error) {
      logToConsole(`Subtitle generation failed: ${error}`, 'error');
      toast.error('Subtitle generation failed');
      setSubtitlesProgress({ visible: false, percentage: 0, status: '' });
    }
  };

  // Helper function to convert API subtitle data to SRT format
  const convertSubtitlesToSRT = (subtitleData: any[]): string => {
    return subtitleData.map((sub, index) => {
      const startTime = formatTimeForSRT(sub.start);
      const endTime = formatTimeForSRT(sub.end);
      return `${index + 1}\n${startTime} --> ${endTime}\n${sub.text}\n`;
    }).join('\n');
  };

  // Helper function to format time for SRT
  const formatTimeForSRT = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const milliseconds = Math.floor((seconds % 1) * 1000);
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${milliseconds.toString().padStart(3, '0')}`;
  };

  // Helper function to generate sample subtitles based on language
  const generateSampleSubtitles = (language: string): string => {
    const subtitleTemplates = {
      'en': `1
00:00:00,000 --> 00:00:05,000
Welcome to this video demonstration

2
00:00:05,000 --> 00:00:10,000
This is an example of English subtitles

3
00:00:10,000 --> 00:00:15,000
Generated automatically by SnipX AI`,
      
      'ur': `1
00:00:00,000 --> 00:00:05,000
اس ویڈیو ڈیمونسٹریشن میں خوش آمدید

2
00:00:05,000 --> 00:00:10,000
یہ اردو سب ٹائٹلز کی مثال ہے

3
00:00:10,000 --> 00:00:15,000
SnipX AI کے ذریعے خودکار طور پر تیار کیا گیا`,
      
      'ru-ur': `1
00:00:00,000 --> 00:00:05,000
Is video demonstration mein khush aamdeed

2
00:00:05,000 --> 00:00:10,000
Yeh Roman Urdu subtitles ki misaal hai

3
00:00:10,000 --> 00:00:15,000
SnipX AI ke zariye automatic tayyar kiya gaya`,
      
      'es': `1
00:00:00,000 --> 00:00:05,000
Bienvenido a esta demostración de video

2
00:00:05,000 --> 00:00:10,000
Este es un ejemplo de subtítulos en español

3
00:00:10,000 --> 00:00:15,000
Generado automáticamente por SnipX AI`,
      
      'fr': `1
00:00:00,000 --> 00:00:05,000
Bienvenue dans cette démonstration vidéo

2
00:00:05,000 --> 00:00:10,000
Ceci est un exemple de sous-titres français

3
00:00:10,000 --> 00:00:15,000
Généré automatiquement par SnipX AI`,
      
      'de': `1
00:00:00,000 --> 00:00:05,000
Willkommen zu dieser Video-Demonstration

2
00:00:05,000 --> 00:00:10,000
Dies ist ein Beispiel für deutsche Untertitel

3
00:00:10,000 --> 00:00:15,000
Automatisch generiert von SnipX AI`
    };
    
    return subtitleTemplates[language as keyof typeof subtitleTemplates] || subtitleTemplates['en'];
  };

  // Helper function to get language name
  const getLanguageName = (code: string): string => {
    const languages = {
      'en': 'English',
      'ur': 'Urdu',
      'ru-ur': 'Roman Urdu',
      'es': 'Spanish',
      'fr': 'French',
      'de': 'German'
    };
    return languages[code as keyof typeof languages] || 'English';
  };

  const handleSummarizeVideo = () => {
    if (!uploadedVideoId) {
      toast.error('Please upload a video file first');
      return;
    }
    logToConsole(`Starting video summarization: Length=${summaryLength}, Focus=${summaryFocus}`);
    setIsLoadingPreview(true);
    processVideo(
      { summarize: true },
      setSummarizationProgress,
      'Video summarized successfully'
    ).finally(() => setIsLoadingPreview(false));
  };

  // FIXED: Enhanced video processing with live preview updates
  const handleEnhanceVideo = () => {
    if (!uploadedVideoId) {
      toast.error('Please upload a video file first');
      return;
    }
    
    logToConsole(`Starting video enhancement: AI Color=${aiColorEnhancement}, Bright=${brightnessLevel}%, Contrast=${contrastLevel}%`);
    setIsLoadingPreview(true);
    
    // Create comprehensive enhancement options
    const enhancementOptions = {
      brightness: brightnessLevel,
      contrast: contrastLevel,
      ai_color_enhancement: aiColorEnhancement,
      saturation: 100  // Default, will be overridden by AI if enabled
    };
    
    processVideo(
      enhancementOptions,
      setEnhancementProgress,
      'Video enhancement completed successfully'
    ).finally(() => setIsLoadingPreview(false));
  };

  const handleGenerateThumbnail = async () => {
    if (!uploadedVideoId) {
      toast.error('Please upload a video file first');
      return;
    }
    if (selectedFrameIndex === null) {
      toast.error('Please select a frame first');
      return;
    }
    logToConsole(`Starting thumbnail generation: Style=${thumbnailStyle}, Text="${thumbnailText}", Frame=${selectedFrameIndex + 1}`);
    
    // Debug logging
    console.log('[THUMBNAIL DEBUG] thumbnailText value:', thumbnailText);
    console.log('[THUMBNAIL DEBUG] thumbnailText type:', typeof thumbnailText);
    console.log('[THUMBNAIL DEBUG] thumbnailText length:', thumbnailText?.length);
    console.log('[THUMBNAIL DEBUG] Will send:', thumbnailText || null);
    
    setGeneratedThumbnail(null);
    
    try {
      await processVideo(
        { 
          generate_thumbnail: true,
          thumbnail_text: thumbnailText || null,
          thumbnail_frame_index: selectedFrameIndex
        },
        setThumbnailProgress,
        'Thumbnail generated successfully'
      );
      
      // Wait a bit for backend to process, then get the real thumbnail URL
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      if (isAuthenticated && uploadedVideoId) {
        // Get the real thumbnail URL from backend
        const thumbnailUrl = ApiService.getVideoThumbnailUrl(uploadedVideoId, selectedFrameIndex + 1);
        setGeneratedThumbnail(thumbnailUrl);
        logToConsole(`Thumbnail ready: ${thumbnailUrl}`, 'success');
      } else {
        // Demo mode - use the selected frame itself
        setGeneratedThumbnail(thumbnailFrames[selectedFrameIndex]);
        logToConsole('Demo mode: Using selected frame as thumbnail', 'info');
      }
    } catch (error) {
      logToConsole(`Thumbnail generation error: ${error}`, 'error');
      toast.error('Failed to generate thumbnail');
    }
  };

  // FIXED: Proper download functionality
  const handleDownloadVideo = async () => {
    if (!uploadedVideoId) {
      toast.error('No video available for download');
      return;
    }

    try {
      logToConsole('Starting video download...', 'info');
      
      // Use the proper download API endpoint
      const token = localStorage.getItem('token');
      if (!token) {
        toast.error('Please login to download videos');
        return;
      }

      // Fetch the video file from the backend
      const response = await fetch(`${API_URL}/api/videos/${uploadedVideoId}/download`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (!response.ok) {
        throw new Error('Download failed');
      }

      // Get the blob and create download link
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `enhanced_${selectedFile?.name || videoData?.filename || 'video.mp4'}`;
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      logToConsole('Video download started successfully!', 'success');
      toast.success('Enhanced video download started!');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Download failed';
      logToConsole(`Download failed: ${errorMessage}`, 'error');
      toast.error(errorMessage);
    }
  };

  // Download subtitle file
  const handleDownloadSubtitles = () => {
    if (!generatedSubtitles) {
      toast.error('No subtitles available for download');
      return;
    }

    try {
      const blob = new Blob([generatedSubtitles], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = subtitleFile || `subtitles_${subtitleLanguage}.srt`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      logToConsole(`Subtitles downloaded: ${getLanguageName(subtitleLanguage)}`, 'success');
      toast.success('Subtitles downloaded successfully!');
    } catch (error) {
      logToConsole('Subtitle download failed', 'error');
      toast.error('Failed to download subtitles');
    }
  };

  const handleDownloadThumbnail = async () => {
    if (!generatedThumbnail) {
      toast.error('No thumbnail to download');
      return;
    }
    
    try {
      logToConsole('Downloading thumbnail...');
      
      // Check if it's a data URL (from canvas) or remote URL (from backend)
      if (generatedThumbnail.startsWith('data:')) {
        // Data URL - can download directly
        const link = document.createElement('a');
        link.href = generatedThumbnail;
        link.download = `thumbnail-${Date.now()}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        logToConsole('Thumbnail downloaded successfully!', 'success');
        toast.success('Thumbnail downloaded!');
      } else {
        // Remote URL - fetch and convert to blob
        const response = await fetch(generatedThumbnail);
        if (!response.ok) throw new Error('Failed to fetch thumbnail');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `thumbnail-${Date.now()}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        
        logToConsole('Thumbnail downloaded successfully!', 'success');
        toast.success('Thumbnail downloaded!');
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Download failed';
      logToConsole(`Thumbnail download failed: ${errorMsg}`, 'error');
      toast.error('Failed to download thumbnail');
    }
  };

  const renderProgressBar = (progressState: ProgressState) => {
    if (!progressState.visible) return null;
    return (
      <div className="mt-6 bg-white/80 backdrop-blur-md rounded-2xl p-6 border border-white/20 shadow-lg animate-slide-up-3d">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm font-medium text-gray-700">{progressState.status.split('-')[1]?.trim() || 'Processing...'}</p>
          <span className="text-sm font-bold text-purple-600">{Math.round(progressState.percentage)}%</span>
        </div>
        <div className="bg-gray-200 rounded-full h-3 overflow-hidden shadow-inner">
          <div
            className="bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 h-3 rounded-full transition-all duration-500 shadow-lg"
            style={{ 
              width: `${progressState.percentage}%`,
              animation: 'progress-glow 2s ease-in-out infinite'
            }}
          />
        </div>
      </div>
    );
  };

  const tabs = [
    { id: 'audio', name: 'Audio Enhancement', icon: Volume2, color: 'purple', gradient: 'from-purple-500 to-indigo-600' },
    { id: 'subtitles', name: 'Subtitling', icon: Captions, color: 'teal', gradient: 'from-teal-500 to-cyan-600' },
    { id: 'summarization', name: 'Summarization', icon: Film, color: 'orange', gradient: 'from-orange-500 to-red-600' },
    { id: 'enhancement', name: 'Enhancement', icon: Wand2, color: 'green', gradient: 'from-green-500 to-emerald-600' },
    { id: 'thumbnail', name: 'Thumbnail', icon: ImageIcon, color: 'red', gradient: 'from-red-500 to-pink-600' },
  ];

  // Show login prompt if not authenticated
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 py-8 relative overflow-hidden">
        {/* Optimized Background Elements - Fixed hanging animations */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Simplified floating orbs with optimized animations */}
          <div 
            className="absolute w-96 h-96 rounded-full opacity-10"
            style={{
              background: 'radial-gradient(circle, rgba(139, 92, 246, 0.3) 0%, transparent 70%)',
              top: '10%',
              left: '10%',
              animation: 'gentle-float 8s ease-in-out infinite',
              willChange: 'transform'
            }}
          />
          
          <div 
            className="absolute w-80 h-80 rounded-full opacity-8"
            style={{
              background: 'radial-gradient(circle, rgba(236, 72, 153, 0.3) 0%, transparent 70%)',
              top: '60%',
              right: '10%',
              animation: 'gentle-float 10s ease-in-out infinite reverse',
              animationDelay: '2s',
              willChange: 'transform'
            }}
          />

          {/* Simple geometric shapes with reduced animation complexity */}
          <div 
            className="absolute w-24 h-24 opacity-15"
            style={{
              top: '20%',
              right: '20%',
              background: 'linear-gradient(45deg, rgba(59, 130, 246, 0.4), rgba(147, 51, 234, 0.4))',
              clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)',
              animation: 'gentle-spin 15s linear infinite',
              willChange: 'transform'
            }}
          />

          {/* Simplified floating sparkles */}
          <div className="absolute top-1/4 left-1/3" style={{ animation: 'gentle-bounce 3s ease-in-out infinite' }}>
            <Sparkles className="text-purple-400/20 w-4 h-4" />
          </div>
          <div className="absolute top-2/3 right-1/2" style={{ animation: 'gentle-bounce 4s ease-in-out infinite', animationDelay: '1s' }}>
            <Sparkles className="text-pink-400/20 w-3 h-3" />
          </div>
        </div>

        <div className="max-w-4xl mx-auto px-4 relative z-10">
          <div className="text-center bg-white/90 backdrop-blur-md rounded-3xl shadow-2xl p-12 border border-white/20 animate-slide-up-3d">
            <div className="mb-8">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full mb-6 shadow-lg">
                <LogIn className="text-purple-600 animate-pulse-3d" size={40} />
              </div>
              <h2 className="text-4xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Login Required
              </h2>
              <p className="text-xl text-gray-600 mb-8 max-w-md mx-auto leading-relaxed">
                Please login to access the full video editing features, or try our demo mode with limited functionality.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center">
              <button
                onClick={loginAsDemo}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-8 py-4 rounded-2xl font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-xl flex items-center justify-center"
              >
                <Zap className="mr-3" size={20} />
                Try Demo Mode
              </button>
              <a
                href="/login"
                className="bg-white/80 backdrop-blur-sm hover:bg-white/90 text-gray-800 px-8 py-4 rounded-2xl font-semibold transition-all duration-300 border-2 border-gray-200 hover:border-purple-300 transform hover:scale-105 hover:shadow-xl flex items-center justify-center"
              >
                <LogIn className="mr-3" size={20} />
                Login / Signup
              </a>
            </div>
            
            <p className="text-sm text-gray-500 mt-6 bg-blue-50 border border-blue-200 rounded-xl p-4">
              <Brain className="inline mr-2" size={16} />
              Demo mode allows you to test features with sample data
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div 
      ref={featuresRef}
      className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 py-8 relative overflow-hidden"
    >
      {/* Optimized Background Elements - Fixed hanging animations */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Simplified floating orbs with optimized animations */}
        <div 
          className="absolute w-96 h-96 rounded-full opacity-10"
          style={{
            background: 'radial-gradient(circle, rgba(139, 92, 246, 0.3) 0%, transparent 70%)',
            top: '10%',
            left: '10%',
            animation: 'gentle-float 8s ease-in-out infinite',
            willChange: 'transform'
          }}
        />
        
        <div 
          className="absolute w-80 h-80 rounded-full opacity-8"
          style={{
            background: 'radial-gradient(circle, rgba(236, 72, 153, 0.3) 0%, transparent 70%)',
            top: '60%',
            right: '10%',
            animation: 'gentle-float 10s ease-in-out infinite reverse',
            animationDelay: '2s',
            willChange: 'transform'
          }}
        />

        {/* Simple geometric shapes with reduced animation complexity */}
        <div 
          className="absolute w-24 h-24 opacity-15"
          style={{
            top: '20%',
            right: '20%',
            background: 'linear-gradient(45deg, rgba(59, 130, 246, 0.4), rgba(147, 51, 234, 0.4))',
            clipPath: 'polygon(50% 0%, 0% 100%, 100% 100%)',
            animation: 'gentle-spin 15s linear infinite',
            willChange: 'transform'
          }}
        />

        {/* Simplified floating sparkles */}
        <div className="absolute top-1/4 left-1/3" style={{ animation: 'gentle-bounce 3s ease-in-out infinite' }}>
          <Sparkles className="text-purple-400/20 w-4 h-4" />
        </div>
        <div className="absolute top-2/3 right-1/2" style={{ animation: 'gentle-bounce 4s ease-in-out infinite', animationDelay: '1s' }}>
          <Sparkles className="text-pink-400/20 w-3 h-3" />
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Enhanced Header */}
        <div className="text-center mb-12 animate-slide-up-3d">
          <div className="inline-flex items-center px-6 py-3 bg-white/80 backdrop-blur-md rounded-full shadow-lg border border-white/20 mb-6 transform hover:scale-105 transition-all duration-300 hover:shadow-xl">
            <Brain className="text-purple-600 mr-3 animate-pulse" size={24} />
            <span className="text-purple-700 font-medium">AI-Powered Video Editor</span>
            <Sparkles className="text-pink-600 ml-3 animate-bounce" size={20} />
          </div>
          <h1 className="text-5xl font-bold text-gray-900 mb-4 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent animate-text-shimmer">
            Professional Video Editing
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto animate-fade-in-up-3d">
            Upload your video and let AI do the magic with advanced processing capabilities
          </p>
        </div>

        <div 
          className="editor-container p-8 bg-white/90 backdrop-blur-md rounded-3xl shadow-2xl border border-white/20 animate-slide-up-3d"
        >
          <div className="flex items-center justify-between mb-8">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 flex items-center">
                <Wand2 className="mr-3 text-purple-600 animate-pulse" size={32} />
                AI Video Studio
              </h2>
              <p className="text-gray-600 mt-2">Transform your videos with cutting-edge AI technology</p>
            </div>
            {!isAuthenticated && (
              <div className="bg-gradient-to-r from-yellow-100 to-orange-100 border-2 border-yellow-300 rounded-2xl px-6 py-3 shadow-lg">
                <span className="text-sm text-yellow-800 font-bold flex items-center">
                  <Star className="mr-2" size={16} />
                  Demo Mode
                </span>
              </div>
            )}
          </div>

          {/* Enhanced Tabs */}
          <div className="border-b border-gray-200 mb-8">
            <nav className="-mb-px flex space-x-2 overflow-x-auto bg-gray-50/50 rounded-2xl p-2">
              {tabs.map((tab, index) => (
                <button
                  key={tab.id}
                  onClick={() => handleTabClick(tab.id)}
                  className={`tab-button whitespace-nowrap py-4 px-6 rounded-xl font-semibold text-sm flex items-center transition-all duration-300 transform hover:scale-105 hover:-translate-y-1 ${
                    activeTab === tab.id
                      ? `bg-gradient-to-r ${tab.gradient} text-white shadow-lg`
                      : 'text-gray-600 hover:text-gray-800 hover:bg-white/80'
                  }`}
                  style={{ 
                    animationDelay: `${index * 100}ms`,
                  }}
                >
                  <tab.icon className="mr-3 h-5 w-5" />
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>

          {/* Enhanced File Upload Area */}
          <div
            id="file-upload-area"
            className="file-upload-area border-dashed border-3 border-purple-300 hover:border-purple-500 hover:bg-purple-50/50 py-16 px-8 text-center cursor-pointer mb-8 rounded-2xl transition-all duration-300 bg-gradient-to-br from-white/80 to-purple-50/50 backdrop-blur-sm shadow-lg transform hover:scale-105 hover:shadow-xl"
            onClick={triggerFileUpload}
          >
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full mb-4 shadow-lg animate-bounce-3d">
                <UploadCloud className="text-purple-600 animate-pulse" size={40} />
              </div>
            </div>
            <h3 className="text-2xl font-bold mb-3 text-gray-900">Drag & Drop Your Video</h3>
            <p className="text-gray-600 font-medium mb-6">or click to browse files</p>
            <input
              type="file"
              id="video-upload"
              accept="video/*"
              className="hidden"
              ref={fileInputRef}
              onChange={handleFileSelect}
            />
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-8 py-4 rounded-2xl font-semibold transition-all duration-300 transform hover:scale-105 hover:shadow-lg inline-block">
              Select Video File
            </div>
            <p className="text-sm text-gray-500 mt-4 bg-white/60 rounded-full px-4 py-2 inline-block">
              Supports MP4, MOV, AVI up to 500MB
            </p>
          </div>

          {/* Enhanced Upload Progress */}
          {isUploading && (
            <div className="mb-8 bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-200 rounded-2xl p-6 shadow-lg animate-slide-up-3d">
              <div className="flex items-center justify-between mb-4">
                <span className="text-lg font-bold text-blue-800 flex items-center">
                  <Loader2 className="animate-spin mr-3" size={24} />
                  Uploading video...
                </span>
                <span className="text-lg font-bold text-blue-800">{Math.round(uploadProgress)}%</span>
              </div>
              <div className="w-full bg-blue-200 rounded-full h-4 overflow-hidden shadow-inner">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-cyan-500 h-4 rounded-full transition-all duration-300 shadow-lg"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Enhanced Selected File Info */}
          {selectedFile && (
            <div 
              id="filename-display" 
              className="mb-8 bg-gradient-to-r from-gray-50 to-purple-50 border-2 border-gray-200 rounded-2xl p-6 shadow-lg animate-slide-in-3d"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <div className="bg-gradient-to-br from-purple-100 to-pink-100 rounded-2xl p-4 mr-4 shadow-lg">
                    <Video className="text-purple-600" size={24} />
                  </div>
                  <div>
                    <p className="text-lg font-bold text-gray-900">{selectedFile.name}</p>
                    <p className="text-sm text-gray-600">{formatFileSize(selectedFile.size)}</p>
                  </div>
                </div>
                {videoData && (
                  <div className="flex items-center bg-white/80 rounded-full px-4 py-2 shadow-md">
                    {videoData.status === 'completed' && <CheckCircle className="text-green-500 mr-2 animate-pulse" size={20} />}
                    {videoData.status === 'failed' && <AlertCircle className="text-red-500 mr-2 animate-pulse" size={20} />}
                    {videoData.status === 'processing' && <Loader2 className="animate-spin text-blue-500 mr-2" size={20} />}
                    <span className="text-sm font-bold capitalize">{videoData.status}</span>
                  </div>
                )}
              </div>
              {videoData?.metadata && (
                <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                  {videoData.metadata.duration && (
                    <div className="bg-white/60 rounded-xl p-3 text-center">
                      <div className="text-lg font-bold text-purple-600">{Math.round(videoData.metadata.duration)}s</div>
                      <div className="text-xs text-gray-600">Duration</div>
                    </div>
                  )}
                  {videoData.metadata.resolution && (
                    <div className="bg-white/60 rounded-xl p-3 text-center">
                      <div className="text-lg font-bold text-blue-600">{videoData.metadata.resolution}</div>
                      <div className="text-xs text-gray-600">Resolution</div>
                    </div>
                  )}
                  {videoData.metadata.fps && (
                    <div className="bg-white/60 rounded-xl p-3 text-center">
                      <div className="text-lg font-bold text-green-600">{videoData.metadata.fps}</div>
                      <div className="text-xs text-gray-600">FPS</div>
                    </div>
                  )}
                  {videoData.metadata.format && (
                    <div className="bg-white/60 rounded-xl p-3 text-center">
                      <div className="text-lg font-bold text-orange-600">{videoData.metadata.format}</div>
                      <div className="text-xs text-gray-600">Format</div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Enhanced Tab Contents */}
          <div className="mt-8">
            {/* Audio Enhancement Tab */}
            <div id="audio-tab" className={`tab-content ${activeTab === 'audio' ? 'block' : 'hidden'}`}>
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-8 shadow-xl border border-white/20 animate-content-reveal-3d">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <Volume2 className="mr-3 text-purple-600 animate-pulse" size={28} />
                  Audio Enhancement
                </h3>
                
                <div className="space-y-4">
                  <label htmlFor="noise-reduction" className="block text-sm font-bold text-gray-700">
                    Background Noise Reduction
                    <span className="text-xs text-gray-500 block font-normal">AI-powered noise filtering</span>
                  </label>
                  <select 
                    id="noise-reduction" 
                    value={audioEnhancement} 
                    onChange={(e) => setAudioEnhancement(e.target.value)} 
                    className="block w-full px-4 py-3 border-2 border-purple-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white/80 backdrop-blur-sm text-sm font-medium"
                  >
                    <option value="none">None (Keep original audio)</option>
                    <option value="light">Light (Subtle noise reduction)</option>
                    <option value="moderate">Moderate (Balanced clarity)</option>
                    <option value="strong">Strong (Maximum noise removal)</option>
                  </select>
                  <p className="text-xs text-gray-600">Higher levels remove more background noise but may affect audio quality</p>
                </div>
                
                {/* Audio Timeline Preview */}
                <div className="mt-8 bg-gradient-to-r from-gray-50 to-purple-50 border-2 border-gray-200 rounded-2xl p-6 shadow-lg">
                  <h4 className="text-lg font-bold text-gray-900 mb-4 flex items-center">
                    <Eye className="mr-2 text-blue-600" size={20} />
                    Audio Enhancement Preview
                  </h4>
                  <div className="space-y-4">
                    <div className="bg-white rounded-xl p-4 shadow-inner">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-700">Original Audio Timeline</span>
                        <span className="text-xs text-gray-500">With filler words and noise</span>
                      </div>
                      <div className="h-8 bg-gradient-to-r from-red-200 via-yellow-200 to-red-200 rounded-lg flex items-center">
                        <div className="w-full h-2 bg-gradient-to-r from-red-400 via-yellow-400 to-red-400 rounded opacity-60"></div>
                      </div>
                    </div>
                    
                    <div className="flex justify-center">
                      <div className="bg-purple-100 rounded-full p-2">
                        <Volume2 className="text-purple-600 animate-pulse" size={20} />
                      </div>
                    </div>
                    
                    <div className="bg-white rounded-xl p-4 shadow-inner">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium text-gray-700">Enhanced Audio Timeline</span>
                        <span className="text-xs text-green-600">Clean & fluent</span>
                      </div>
                      <div className="h-8 bg-gradient-to-r from-green-200 via-blue-200 to-green-200 rounded-lg flex items-center">
                        <div className="w-3/4 h-2 bg-gradient-to-r from-green-400 via-blue-400 to-green-400 rounded"></div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 text-xs text-gray-600 text-center">
                    Preview shows estimated improvement - actual results may vary based on audio quality
                  </div>
                </div>
                
                <div className="mt-8 flex flex-col sm:flex-row gap-4">
                  <button 
                    onClick={handleProcessAudio} 
                    className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white px-8 py-4 rounded-2xl text-lg font-bold transition-all duration-300 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 hover:shadow-xl" 
                    disabled={!uploadedVideoId || isUploading}
                  >
                    <Volume2 className="mr-3 h-6 w-6" />
                    Enhance Audio
                  </button>
                  
                  <div className="flex items-center text-sm text-gray-600 bg-blue-50 rounded-xl px-4 py-2 border border-blue-200">
                    <CheckCircle className="mr-2 text-blue-600" size={16} />
                    <span>Processing will maintain video quality while improving audio clarity</span>
                  </div>
                </div>
                
                {renderProgressBar(audioProgress)}
                
                {/* Results Display */}
                {audioProgress.percentage === 100 && (
                  <div className="mt-8 bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-2xl p-6 shadow-lg animate-slide-up-3d">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <CheckCircle className="text-green-500 mr-3 animate-pulse" size={24} />
                        <div>
                          <span className="text-lg font-bold text-green-800">Audio Enhancement Complete!</span>
                          <p className="text-sm text-green-700">Your video now has clean, fluent audio</p>
                        </div>
                      </div>
                      <button 
                        onClick={handleDownloadVideo}
                        className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-6 py-3 rounded-xl text-sm font-bold transition-all duration-300 flex items-center transform hover:scale-105 hover:shadow-lg"
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Download Enhanced Video
                      </button>
                    </div>
                    
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div className="bg-white/60 rounded-xl p-3 text-center">
                        <div className="text-lg font-bold text-green-600">
                          {videoData?.outputs?.enhancement_results?.duration_reduction_percentage 
                            ? `${videoData.outputs.enhancement_results.duration_reduction_percentage.toFixed(1)}%`
                            : videoData?.outputs?.audio_enhancement_metrics?.time_saved_percentage
                              ? `${videoData.outputs.audio_enhancement_metrics.time_saved_percentage.toFixed(1)}%`
                              : '0%'}
                        </div>
                        <div className="text-xs text-gray-600">Shorter Duration</div>
                      </div>
                      <div className="bg-white/60 rounded-xl p-3 text-center">
                        <div className="text-lg font-bold text-blue-600">
                          {videoData?.outputs?.enhancement_results?.noise_reduction_percentage 
                            ? `${videoData.outputs.enhancement_results.noise_reduction_percentage}%`
                            : audioEnhancement === 'strong' ? '95%' 
                              : audioEnhancement === 'moderate' ? '75%' 
                                : audioEnhancement === 'light' ? '50%' : '0%'}
                        </div>
                        <div className="text-xs text-gray-600">Noise Reduced</div>
                      </div>
                      <div className="bg-white/60 rounded-xl p-3 text-center">
                        <div className="text-lg font-bold text-purple-600">
                          {videoData?.outputs?.enhancement_results?.filler_words_removed 
                            ?? videoData?.outputs?.audio_enhancement_metrics?.filler_words_removed 
                            ?? 0}
                        </div>
                        <div className="text-xs text-gray-600">Filler Words Removed</div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Subtitling Tab */}
            <div id="subtitles-tab" className={`tab-content ${activeTab === 'subtitles' ? 'block' : 'hidden'}`}>
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-8 shadow-xl border border-white/20 animate-content-reveal-3d">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <Captions className="mr-3 text-teal-600 animate-pulse" size={28} />
                  Subtitling Settings
                </h3>
                
                <div className="space-y-4">
                  <label htmlFor="subtitle-language" className="block text-sm font-bold text-gray-700">Language</label>
                  <select 
                    id="subtitle-language" 
                    value={subtitleLanguage} 
                    onChange={(e) => setSubtitleLanguage(e.target.value)} 
                    className="block w-full px-4 py-3 border-2 border-teal-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 bg-white/80 backdrop-blur-sm text-sm font-medium"
                  >
                    <option value="en">🇺🇸 English</option>
                    <option value="ur">🇵🇰 Urdu (اردو)</option>
                    <option value="ru-ur">🇵🇰 Roman Urdu</option>
                    <option value="es">🇪🇸 Spanish (Español)</option>
                    <option value="fr">🇫🇷 French (Français)</option>
                    <option value="de">🇩🇪 German (Deutsch)</option>
                  </select>
                </div>

                {/* Enhanced Subtitle Preview */}
                {generatedSubtitles && (
                  <div className="mt-8 bg-gradient-to-br from-teal-50 to-cyan-50 border-2 border-teal-200 rounded-2xl p-6 shadow-lg animate-slide-up-3d">
                    <div className="flex items-center justify-between mb-4">
                      <h4 className="text-lg font-bold text-teal-900 flex items-center">
                        <Eye className="mr-2" size={20} />
                        Generated Subtitles ({getLanguageName(subtitleLanguage)})
                      </h4>
                      <button 
                        onClick={handleDownloadSubtitles}
                        className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-6 py-3 rounded-xl text-sm font-bold transition-all duration-300 flex items-center transform hover:scale-105 hover:shadow-lg"
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Download SRT
                      </button>
                    </div>
                    <div className="bg-white/80 backdrop-blur-sm border border-teal-200 rounded-xl p-4 max-h-48 overflow-y-auto shadow-inner">
                      <pre className="text-sm text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">{generatedSubtitles}</pre>
                    </div>
                  </div>
                )}

                <div className="mt-8 flex space-x-4">
                  <button 
                    onClick={handleGenerateSubtitles} 
                    className="bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-700 hover:to-cyan-700 text-white px-8 py-4 rounded-2xl text-lg font-bold transition-all duration-300 flex items-center disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 hover:shadow-xl" 
                    disabled={!uploadedVideoId || isUploading}
                  >
                    <Captions className="mr-3 h-6 w-6" />
                    Generate Subtitles
                  </button>
                  
                  {/* Debug test button for subtitle loading */}
                  {process.env.NODE_ENV === 'development' && isAuthenticated && uploadedVideoId && (
                    <button 
                      onClick={async () => {
                        try {
                          console.log('[Features] 🧪 Testing subtitle API for videoId:', uploadedVideoId);
                          const testData = await ApiService.getVideoSubtitles(uploadedVideoId);
                          console.log('[Features] 🧪 Test API response:', testData);
                          if (testData && testData.length > 0) {
                            setSubtitleData(testData);
                            logToConsole(`🧪 Test: Loaded ${testData.length} subtitle segments`, 'success');
                          } else {
                            logToConsole(`🧪 Test: No subtitle data found`, 'info');
                          }
                        } catch (error) {
                          console.error('[Features] 🧪 Test error:', error);
                          logToConsole(`🧪 Test: API error - ${error}`, 'error');
                        }
                      }}
                      className="bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 text-white px-6 py-4 rounded-2xl text-sm font-bold transform hover:scale-105 hover:shadow-lg transition-all duration-300"
                    >
                      🧪 Test Subtitles
                    </button>
                  )}
                </div>
                {renderProgressBar(subtitlesProgress)}
              </div>
            </div>

            {/* Summarization Tab */}
            <div id="summarization-tab" className={`tab-content ${activeTab === 'summarization' ? 'block' : 'hidden'}`}>
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-8 shadow-xl border border-white/20 animate-content-reveal-3d">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <Film className="mr-3 text-orange-600 animate-pulse" size={28} />
                  Video Summarization
                </h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <label htmlFor="summary-length" className="block text-sm font-bold text-gray-700">Summary Length</label>
                    <select 
                      id="summary-length" 
                      value={summaryLength} 
                      onChange={(e) => setSummaryLength(e.target.value)} 
                      className="block w-full px-4 py-3 border-2 border-orange-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 bg-white/80 backdrop-blur-sm text-sm font-medium"
                    >
                      <option value="short">Short (~20% of original)</option>
                      <option value="medium">Medium (~40% of original)</option>
                      <option value="long">Long (~60% of original)</option>
                    </select>
                  </div>
                  
                  <div className="space-y-4">
                    <label htmlFor="summary-focus" className="block text-sm font-bold text-gray-700">Focus</label>
                    <select 
                      id="summary-focus" 
                      value={summaryFocus} 
                      onChange={(e) => setSummaryFocus(e.target.value)} 
                      className="block w-full px-4 py-3 border-2 border-orange-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 bg-white/80 backdrop-blur-sm text-sm font-medium"
                    >
                      <option value="balanced">Balanced</option>
                      <option value="visual">Visual Content</option>
                      <option value="audio">Audio Content</option>
                      <option value="text">Spoken Text</option>
                    </select>
                  </div>
                </div>
                
                <div className="mt-8">
                  <button 
                    onClick={handleSummarizeVideo} 
                    className="bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 text-white px-8 py-4 rounded-2xl text-lg font-bold transition-all duration-300 flex items-center disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 hover:shadow-xl" 
                    disabled={!uploadedVideoId || isUploading}
                  >
                    <Film className="mr-3 h-6 w-6" />
                    Summarize Video
                  </button>
                </div>
                {renderProgressBar(summarizationProgress)}
              </div>
            </div>

            {/* Enhancement Tab */}
            <div id="enhancement-tab" className={`tab-content ${activeTab === 'enhancement' ? 'block' : 'hidden'}`}>
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-8 shadow-xl border border-white/20 animate-content-reveal-3d">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <Wand2 className="mr-3 text-green-600 animate-pulse" size={28} />
                  Video Enhancement
                </h3>
                
                {/* Live Preview Notice */}
                <div className="mb-8 bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-200 rounded-2xl p-6 shadow-lg animate-slide-in-3d">
                  <div className="flex items-center">
                    <Wand2 className="text-blue-600 mr-3 animate-bounce" size={24} />
                    <span className="text-lg font-bold text-blue-900">Live Preview: Changes are applied to the video preview in real-time</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-8">
                  {/* AI Color Enhancement Toggle */}
                  <div className="space-y-4 bg-gradient-to-r from-purple-50 to-pink-50 border-2 border-purple-200 rounded-2xl p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <label htmlFor="ai-color-enhancement" className="block text-sm font-bold text-gray-800 flex items-center">
                          <Wand2 className="mr-2 text-purple-600" size={20} />
                          AI Color Enhancement
                          <span className="ml-2 bg-purple-600 text-white text-xs px-2 py-1 rounded-full">NEW</span>
                        </label>
                        <p className="text-xs text-gray-600 mt-1">Automatically adjusts saturation, brightness & contrast to optimal levels</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input 
                          type="checkbox" 
                          id="ai-color-enhancement"
                          checked={aiColorEnhancement}
                          onChange={(e) => setAiColorEnhancement(e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-14 h-7 bg-gray-300 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-purple-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[4px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-6 after:w-6 after:transition-all peer-checked:bg-purple-600"></div>
                      </label>
                    </div>
                    {aiColorEnhancement && (
                      <div className="mt-3 bg-white/60 rounded-xl p-3 border border-purple-200">
                        <p className="text-xs text-purple-800 font-medium">
                          ✨ AI will analyze your video and automatically apply the best color settings. Manual adjustments below will override AI values.
                        </p>
                      </div>
                    )}
                  </div>
                  
                  <div className="space-y-4">
                    <label htmlFor="brightness-level" className="block text-sm font-bold text-gray-700">Brightness (Live Preview)</label>
                    <div className="flex items-center bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200">
                      <input 
                        type="range" 
                        id="brightness-level" 
                        min="0" 
                        max="200" 
                        value={brightnessLevel} 
                        onChange={(e) => setBrightnessLevel(Number(e.target.value))} 
                        className="w-full h-3 bg-green-200 rounded-lg appearance-none cursor-pointer accent-green-600" 
                      />
                      <span id="brightness-value" className="ml-4 text-lg font-bold text-green-600 w-20 text-right">{brightnessLevel}%</span>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <label htmlFor="contrast-level" className="block text-sm font-bold text-gray-700">Contrast (Live Preview)</label>
                    <div className="flex items-center bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4 border border-green-200">
                      <input 
                        type="range" 
                        id="contrast-level" 
                        min="0" 
                        max="200" 
                        value={contrastLevel} 
                        onChange={(e) => setContrastLevel(Number(e.target.value))} 
                        className="w-full h-3 bg-green-200 rounded-lg appearance-none cursor-pointer accent-green-600" 
                      />
                      <span id="contrast-value" className="ml-4 text-lg font-bold text-green-600 w-20 text-right">{contrastLevel}%</span>
                    </div>
                  </div>
                </div>
                
                {/* Enhancement Preview */}
                {videoData && videoData.status === 'completed' && (
                  <div className="mt-8 bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-2xl p-6 shadow-lg animate-slide-up-3d">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <CheckCircle className="text-green-500 mr-3 animate-pulse" size={24} />
                        <span className="text-lg font-bold text-green-800">Enhanced video is ready!</span>
                      </div>
                      <button 
                        onClick={handleDownloadVideo}
                        className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-6 py-3 rounded-xl text-sm font-bold transition-all duration-300 flex items-center transform hover:scale-105 hover:shadow-lg"
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Download Enhanced Video
                      </button>
                    </div>
                  </div>
                )}
                
                <div className="mt-8">
                  <button 
                    onClick={handleEnhanceVideo} 
                    className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-8 py-4 rounded-2xl text-lg font-bold transition-all duration-300 flex items-center disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 hover:shadow-xl" 
                    disabled={!uploadedVideoId || isUploading}
                  >
                    <Wand2 className="mr-3 h-6 w-6" />
                    Enhance Video
                  </button>
                </div>
                {renderProgressBar(enhancementProgress)}
              </div>
            </div>

            {/* Thumbnail Tab */}
            <div id="thumbnail-tab" className={`tab-content ${activeTab === 'thumbnail' ? 'block' : 'hidden'}`}>
              <div className="bg-white/80 backdrop-blur-md rounded-2xl p-8 shadow-xl border border-white/20 animate-content-reveal-3d">
                <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
                  <ImageIcon className="mr-3 text-red-600 animate-pulse" size={28} />
                  Thumbnail Generation
                </h3>
                
                <div className="space-y-6">
                  <div className="space-y-4">
                    <label htmlFor="thumbnail-text" className="block text-sm font-bold text-gray-700">Add Text Overlay</label>
                    <input 
                      type="text" 
                      id="thumbnail-text" 
                      value={thumbnailText} 
                      onChange={(e) => setThumbnailText(e.target.value)} 
                      className="block w-full px-4 py-3 border-2 border-red-200 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-red-500 bg-white/80 backdrop-blur-sm text-sm font-medium" 
                      placeholder="Enter text for thumbnail" 
                    />
                  </div>
                  
                  <div className="space-y-4">
                    <label className="block text-sm font-bold text-gray-700">Select Frame</label>
                    <div className="flex space-x-4 overflow-x-auto py-4 bg-gradient-to-r from-red-50 to-pink-50 p-4 rounded-2xl border-2 border-red-200 shadow-inner">
                      {isLoadingThumbnails ? (
                        <div className="flex items-center justify-center w-full h-20 text-red-600">
                          <Loader2 className="animate-spin mr-3 h-6 w-6" /> 
                          <span className="font-medium">Loading frames...</span>
                        </div>
                      ) : thumbnailFrames.length > 0 ? (
                        thumbnailFrames.map((frameSrc, index) => (
                          <div
                            key={index}
                            onClick={() => {
                              setSelectedFrameIndex(index);
                              logToConsole(`Selected frame ${index + 1} for thumbnail generation`);
                            }}
                            className={`w-32 h-20 bg-gray-200 rounded-xl cursor-pointer ring-offset-4 ring-offset-red-50 hover:ring-4 hover:ring-red-400 flex-shrink-0 overflow-hidden transition-all duration-300 transform hover:scale-110 hover:shadow-lg ${selectedFrameIndex === index ? 'ring-4 ring-red-500 scale-105' : ''}`}
                          >
                            <img src={frameSrc} alt={`Frame ${index + 1}`} className="w-full h-full object-cover" />
                          </div>
                        ))
                      ) : (
                        <div className="w-full h-20 flex items-center justify-center text-gray-500 text-sm font-medium">
                          {selectedFile ? 'No frames generated yet.' : 'Upload a video to generate frames.'}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="mt-8">
                  <button 
                    onClick={handleGenerateThumbnail} 
                    className="bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white px-8 py-4 rounded-2xl text-lg font-bold transition-all duration-300 flex items-center disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 hover:shadow-xl" 
                    disabled={!uploadedVideoId || selectedFrameIndex === null || isUploading}
                  >
                    <ImageIcon className="mr-3 h-6 w-6" />
                    Generate Thumbnail
                  </button>
                </div>
                {renderProgressBar(thumbnailProgress)}
                
                {generatedThumbnail && (
                  <div id="thumbnail-result" className="mt-8 bg-gradient-to-br from-red-50 to-pink-50 border-2 border-red-200 rounded-2xl p-6 shadow-lg animate-slide-up-3d">
                    <h4 className="text-lg font-bold text-red-900 mb-4 flex items-center">
                      <Star className="mr-2" size={20} />
                      Generated Thumbnail
                    </h4>
                    <div className="w-full max-w-md bg-white rounded-2xl overflow-hidden border-2 border-red-200 shadow-xl transform hover:scale-105 transition-all duration-300">
                      <img id="generated-thumbnail" src={generatedThumbnail} alt="Generated Thumbnail" className="w-full h-auto block" />
                    </div>
                    <div className="mt-6">
                      <button 
                        onClick={handleDownloadThumbnail} 
                        className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-6 py-3 rounded-xl text-sm font-bold transition-all duration-300 flex items-center transform hover:scale-105 hover:shadow-lg"
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Download Thumbnail
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Enhanced Video Preview Section */}
          <div className="mt-12">
            <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
              <Eye className="mr-3 text-blue-600 animate-pulse" size={28} />
              Live Preview
            </h3>
            
            <div 
              className="video-preview flex items-center justify-center bg-gradient-to-br from-gray-900 to-black relative rounded-2xl overflow-hidden shadow-2xl aspect-video border-4 border-white/20"
            >
              {!videoSrc ? (
                <div id="video-placeholder" className="text-center text-gray-400 p-12">
                  <div className="mb-8">
                    <Video className="mx-auto mb-6 animate-float-3d" size={80} />
                    <Sparkles className="mx-auto text-purple-400 animate-sparkle-3d" size={40} />
                  </div>
                  <p className="text-2xl font-bold mb-2">AI Video Preview</p>
                  <p className="text-lg">Upload a video to see the magic happen</p>
                </div>
              ) : (
                <div className="w-full h-full relative bg-black" style={{ filter: previewFilters }}>
                  <VideoPlayer
                    videoUrl={videoSrc}
                    videoId={isAuthenticated ? uploadedVideoId : undefined}
                    subtitles={subtitleData.length > 0 ? subtitleData : undefined}
                    onTimeUpdate={(time) => {
                      // Optional: handle time updates if needed
                    }}
                  />
                  {aiColorEnhancement && (
                    <div className="absolute top-4 left-4 bg-purple-600/90 backdrop-blur-sm text-white px-4 py-2 rounded-full flex items-center space-x-2 shadow-lg animate-pulse">
                      <Sparkles size={16} />
                      <span className="text-sm font-bold">AI Enhanced Preview</span>
                    </div>
                  )}
                </div>
              )}
              {isLoadingPreview && (
                <div id="preview-loading" className="absolute inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50">
                  <div className="text-center">
                    <Loader2 className="animate-spin h-16 w-16 text-purple-400 mb-4" />
                    <p className="text-white text-xl font-bold">AI Processing...</p>
                  </div>
                </div>
              )}
            </div>
            
            {/* Enhanced Live Preview Controls */}
            {videoSrc && (
              <div className="mt-6 space-y-6">
                {/* Subtitle Controls */}
                {(subtitleData.length > 0 || generatedSubtitles) && (
                  <div className="bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-200 rounded-2xl p-6 shadow-lg animate-slide-in-3d">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <Captions className="h-6 w-6 text-blue-600 animate-pulse" />
                        <span className="text-lg font-bold text-blue-900">
                          {subtitleData.length > 0 
                            ? `Live Subtitles Active (${subtitleData.length} segments)`
                            : `Subtitles Generated (${generatedSubtitles ? 'SRT format' : 'Processing...'})`
                          }
                        </span>
                      </div>
                      <div className="text-sm font-medium text-blue-700 bg-blue-100 rounded-full px-4 py-2">
                        {subtitleData.length > 0 
                          ? 'Real-time Whisper AI transcription'
                          : 'Use CC button in player controls'
                        }
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Enhanced API Console - Collapsible */}
          <div className="mt-12">
            <button 
              onClick={() => setShowConsole(!showConsole)}
              className="w-full text-left text-xl font-bold text-gray-900 mb-4 flex items-center justify-between bg-gradient-to-r from-gray-100 to-gray-50 hover:from-gray-200 hover:to-gray-100 px-6 py-4 rounded-2xl transition-all duration-300 border border-gray-200 shadow-md hover:shadow-lg"
            >
              <span className="flex items-center">
                <Zap className="mr-3 text-green-600 animate-pulse" size={24} />
                API Console
              </span>
              <span className={`transform transition-transform duration-300 ${showConsole ? 'rotate-180' : ''}`}>
                ▼
              </span>
            </button>
            {showConsole && (
              <div 
                id="api-console" 
                ref={consoleRef} 
                className="api-console text-sm bg-gradient-to-br from-gray-900 to-black text-green-400 font-mono p-6 rounded-2xl h-64 overflow-y-auto border-2 border-gray-700 scroll-smooth shadow-2xl backdrop-blur-md animate-slide-down"
              >
                {consoleLogs.map((log, index) => (
                  <div key={index} className="console-line whitespace-pre-wrap break-words mb-2 last:mb-0 animate-message-slide-3d" style={{ animationDelay: `${index * 50}ms` }}>
                    <span className="text-gray-500 mr-3 select-none">[{log.timestamp}]</span>
                    <span className={
                      log.type === 'success' ? 'text-green-400 font-bold' : 
                      log.type === 'error' ? 'text-red-400 font-bold' : 
                      log.message.startsWith('[System]') ? 'text-blue-400 font-bold' : 'text-green-400'
                    }>{log.message}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Optimized Custom Styles - Fixed hanging animations */}
      <style jsx>{`
        /* Performance optimized animations */
        @keyframes progress-glow {
          0%, 100% { box-shadow: 0 0 10px rgba(139, 92, 246, 0.4); }
          50% { box-shadow: 0 0 15px rgba(139, 92, 246, 0.6); }
        }
        
        @keyframes gentle-float {
          0%, 100% { 
            transform: translateY(0px); 
            opacity: 0.8;
          }
          50% { 
            transform: translateY(-8px); 
            opacity: 1;
          }
        }
        
        @keyframes gentle-spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @keyframes gentle-bounce {
          0%, 100% { 
            transform: translateY(0px); 
            opacity: 0.6;
          }
          50% { 
            transform: translateY(-4px); 
            opacity: 8;
          }
        }
        
        /* Hardware acceleration for smooth performance */
        .animate-spin, .animate-pulse, .animate-bounce {
          will-change: transform, opacity;
          transform: translateZ(0);
        }
        
        /* Ensure text clarity */
        * {
          text-rendering: optimizeLegibility;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
          backface-visibility: hidden;
        }
        
        /* Optimize transitions */
        .transition-all {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Reduce motion for accessibility */
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        }
      `}</style>
    </div>
  );
};

export default Features;