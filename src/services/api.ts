import { z } from 'zod';

const API_URL = 'http://localhost:5001/api';

// Interface for subtitle data
export interface SubtitleData {
  id: number;
  start: number;
  end: number;
  text: string;
  language: string;
  style: string;
}

// Validation schemas
const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8)
});

const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  firstName: z.string().min(2),
  lastName: z.string().min(2)
});

const videoOptionsSchema = z.object({
  cut_silence: z.boolean().optional(),
  enhance_audio: z.boolean().optional(),
  generate_thumbnail: z.boolean().optional(),
  generate_subtitles: z.boolean().optional(),
  summarize: z.boolean().optional(),
  // Enhancement specific options
  stabilization: z.string().optional(),
  audio_enhancement_type: z.string().optional(),
  pause_threshold: z.number().optional(),
  noise_reduction: z.string().optional(),
  brightness: z.number().optional(),
  contrast: z.number().optional(),
  // Subtitle specific options
  subtitle_language: z.string().optional(),
  subtitle_style: z.string().optional(),
  // Thumbnail specific options
  thumbnail_text: z.string().nullable().optional(),
  thumbnail_frame_index: z.number().nullable().optional()
});

export class ApiService {
  private static token: string | null = null;

  static setToken(token: string) {
    this.token = token;
    localStorage.setItem('token', token);
  }

  static getToken(): string | null {
    if (!this.token) {
      this.token = localStorage.getItem('token');
    }
    return this.token;
  }

  static clearToken() {
    this.token = null;
    localStorage.removeItem('token');
  }

  private static async request(endpoint: string, options: RequestInit = {}) {
    const token = this.getToken();
    const isForm = options.body instanceof FormData;

    const headers: HeadersInit = {
      // Only set JSON content-type when not sending FormData
      ...(isForm ? {} : { 'Content-Type': 'application/json' }),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers
    };

    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers
      });

      if (!res.ok) {
        // Try to parse JSON error, fallback to text
        let msg = `HTTP ${res.status}`;
        try {
          const payload = await res.json();
          msg = payload.message || payload.error || msg;
        } catch {
          msg = await res.text();
        }
        throw new Error(msg);
      }

      // Handle no-content responses
      if (res.status === 204) {
        return null;
      }

      return await res.json();
    } catch (err) {
      console.error('API request failed:', err);
      throw err;
    }
  }

  static async login(email: string, password: string) {
    const validated = loginSchema.parse({ email, password });
    const data = await this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify(validated)
    });
    this.setToken(data.token);
    return data;
  }

  static async register(data: {
    email: string;
    password: string;
    firstName: string;
    lastName: string;
  }) {
    const validated = registerSchema.parse(data);
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify(validated)
    });
  }

  static async uploadVideo(file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append('video', file);
    
    // If progress callback is provided, simulate progress
    if (onProgress) {
      const interval = setInterval(() => {
        onProgress(Math.min(90, Math.random() * 85 + 10));
      }, 200);
      
      try {
        const result = await this.request('/upload', {
          method: 'POST',
          body: formData
        });
        clearInterval(interval);
        onProgress(100);
        return result;
      } catch (error) {
        clearInterval(interval);
        throw error;
      }
    }
    
    return this.request('/upload', {
      method: 'POST',
      body: formData
    });
  }

  static async processVideo(videoId: string, options: {
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
    // Thumbnail specific options
    thumbnail_text?: string | null;
    thumbnail_frame_index?: number | null;
  }) {
    const validated = videoOptionsSchema.parse(options);
    return this.request(`/videos/${videoId}/process`, {
      method: 'POST',
      body: JSON.stringify({ options: validated })
    });
  }

  static async getVideoStatus(videoId: string) {
    return this.request(`/videos/${videoId}`);
  }

  static async getVideo(videoId: string) {
    return this.getVideoStatus(videoId);
  }

  static async getUserVideos() {
    return this.request('/videos');
  }

  static async deleteVideo(videoId: string) {
    return this.request(`/videos/${videoId}`, {
      method: 'DELETE'
    });
  }

  // Enhanced subtitle generation with Whisper
  static async generateAdvancedSubtitles(videoId: string, options: {
    language: string;
    style: string;
    format?: 'srt' | 'json' | 'both';
  }) {
    return this.request(`/videos/${videoId}/subtitles/advanced`, {
      method: 'POST',
      body: JSON.stringify(options)
    });
  }

  // Real-time thumbnail generation
  static async generateThumbnails(videoId: string, options: {
    count?: number;
    timestamps?: number[];
    style?: 'auto' | 'manual';
  } = {}) {
    return this.request(`/videos/${videoId}/thumbnails/generate`, {
      method: 'POST',
      body: JSON.stringify(options)
    });
  }

  // Real-time audio enhancement
  static async enhanceAudio(videoId: string, options: {
    type?: 'clear' | 'music' | 'full';
    noiseReduction?: boolean;
    volumeBoost?: number;
    equalization?: boolean;
  } = {}) {
    return this.request(`/videos/${videoId}/audio/enhance`, {
      method: 'POST',
      body: JSON.stringify(options)
    });
  }

  // Get processing status with real-time updates
  static async getProcessingStatus(videoId: string) {
    return this.request(`/videos/${videoId}/status`);
  }

  // Get subtitle data in JSON format for live preview
  static async getSubtitleData(videoId: string, language: string) {
    return this.request(`/videos/${videoId}/subtitles/${language}/json`);
  }

  // Get video subtitles for player
  static async getVideoSubtitles(videoId: string): Promise<SubtitleData[]> {
    try {
      const response = await this.request(`/videos/${videoId}/subtitles`);
      console.log('[ApiService] getVideoSubtitles response:', response);
      
      // Handle different response formats
      if (Array.isArray(response)) {
        return response;
      } else if (response?.segments && Array.isArray(response.segments)) {
        return response.segments;
      } else if (response?.subtitles?.segments && Array.isArray(response.subtitles.segments)) {
        return response.subtitles.segments;
      } else {
        console.log('[ApiService] No valid subtitle segments found');
        return [];
      }
    } catch (error) {
      console.error('[ApiService] Error fetching subtitles:', error);
      return [];
    }
  }

  // Generate subtitles for a video
  static async generateSubtitles(videoId: string, options: {
    language: string;
    style: string;
  }) {
    return this.request(`/videos/${videoId}/subtitles/generate`, {
      method: 'POST',
      body: JSON.stringify(options)
    });
  }

  // Profile management
  static async getUserProfile() {
    return this.request('/profile');
  }

  static async updateUserProfile(data: {
    firstName?: string;
    lastName?: string;
    email?: string;
    preferences?: Record<string, any>;
  }) {
    return this.request('/profile', {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }

  // Admin endpoints
  static async getAdminStats() {
    return this.request('/admin/stats');
  }

  static async getAllUsers() {
    return this.request('/admin/users');
  }

  static async getAllVideos() {
    return this.request('/admin/videos');
  }

  static async updateUserStatus(userId: string, status: string) {
    return this.request(`/admin/users/${userId}/status`, {
      method: 'PUT',
      body: JSON.stringify({ status })
    });
  }

  // Support ticket
  static async submitSupportTicket(data: {
    name: string;
    email: string;
    subject: string;
    description: string;
    priority: string;
    type: string;
  }) {
    return this.request('/support/tickets', {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }

  // Export video with edits (trim, text overlay, audio settings)
  static async exportVideo(videoId: string, options: {
    trim_start?: number;
    trim_end?: number;
    text_overlay?: string;
    text_position?: string;
    text_color?: string;
    text_size?: number;
    music_volume?: number;
    video_volume?: number;
    mute_original?: boolean;
  }) {
    return this.request(`/videos/${videoId}/export`, {
      method: 'POST',
      body: JSON.stringify(options)
    });
  }

  // Download exported video
  static async downloadExportedVideo(videoId: string): Promise<Blob> {
    const token = this.getToken();
    const response = await fetch(`${API_URL}/videos/${videoId}/download-export`, {
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      }
    });

    if (!response.ok) {
      throw new Error('Download failed');
    }

    return response.blob();
  }

  // Download processed video
  static async downloadVideo(videoId: string): Promise<Blob> {
    const token = this.getToken();
    const response = await fetch(`${API_URL}/videos/${videoId}/download`, {
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      }
    });

    if (!response.ok) {
      throw new Error('Download failed');
    }

    return response.blob();
  }

  // Download subtitle file
  static async downloadSubtitles(videoId: string, language: string, format: 'srt' | 'json' = 'srt'): Promise<Blob> {
    const token = this.getToken();
    const response = await fetch(`${API_URL}/videos/${videoId}/subtitles/${language}/download?format=${format}`, {
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {})
      }
    });

    if (!response.ok) {
      throw new Error('Subtitle download failed');
    }

    return response.blob();
  }

  // Delete user account
  static async deleteAccount() {
    return this.request('/auth/delete-account', {
      method: 'DELETE'
    });
  }

  // Get video thumbnail URL
  static getVideoThumbnailUrl(videoId: string, index?: number): string {
    const token = this.getToken();
    const indexParam = index !== undefined ? `index=${index}` : '';
    const tokenParam = token ? `token=${encodeURIComponent(token)}` : '';
    const separator = indexParam && tokenParam ? '&' : '';
    const queryString = indexParam || tokenParam ? `?${indexParam}${separator}${tokenParam}` : '';
    return `${API_URL}/videos/${videoId}/thumbnail${queryString}`;
  }

  // Get all thumbnails for a video
  static async getVideoThumbnails(videoId: string) {
    return this.request(`/videos/${videoId}/thumbnails`);
  }
}