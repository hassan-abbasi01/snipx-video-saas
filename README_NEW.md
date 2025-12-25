# 🎬 SnipX - AI-Powered Video Processing Platform

## 📋 Project Overview (Project Ki Tafseel)

**SnipX** ek complete AI video processing platform hai jo users ko videos upload, process, aur edit karne ki facility deta hai. Is project mein powerful AI models use hoti hain jo automatic subtitles generate karte hain, audio enhance karte hain, aur thumbnails create karte hain.

### Main Features (Mukhya Features):
✅ **User Management** - Registration, Login, Email Verification  
✅ **Video Upload & Processing** - AI-powered video processing  
✅ **Auto Subtitle Generation** - Multiple languages support (Urdu, English, Arabic, etc.)  
✅ **Audio Enhancement** - Noise reduction using AI  
✅ **AI Thumbnail Generation** - Automatic thumbnail creation  
✅ **Admin Dashboard** - Complete monitoring system  
✅ **Support Ticket System** - User support management  

---

## 🛠️ Technology Stack (Istemal Hone Wali Technologies)

### Frontend (User Interface):
```
Language:    TypeScript (Type-safe JavaScript)
Framework:   React 18 (UI components banane ke liye)
Build Tool:  Vite 6.0 (Fast development server)
Styling:     Tailwind CSS (Utility-first CSS framework)
Charts:      Chart.js + react-chartjs-2 (Data visualization)
HTTP Client: Axios (API calls ke liye)
Routing:     React Router v6 (Page navigation)
State:       Context API (Global state management)
Port:        5173 (Development server)
```

**Kyun istemal kiya:**
- **TypeScript**: Code mein errors kam aate hain, autocomplete milta hai
- **React**: Component-based architecture, fast rendering
- **Vite**: Webpack se zyada fast, instant HMR (Hot Module Replacement)
- **Tailwind**: Styling bohot fast, responsive design easy
- **Chart.js**: Admin dashboard ke charts ke liye

### Backend (Server-Side):
```
Language:    Python 3.12
Framework:   Flask 3.0.0 (Lightweight web framework)
Database:    MongoDB (NoSQL database)
AI/ML:       OpenAI Whisper (Speech-to-text)
             noisereduce (Audio noise removal)
             TensorFlow (AI model loading)
Video:       FFmpeg (Video processing)
             OpenCV (Computer vision tasks)
Auth:        JWT (JSON Web Tokens - secure authentication)
Password:    bcrypt (Password hashing)
Port:        5001 (Production server)
```

**Kyun istemal kiya:**
- **Python**: AI/ML libraries ka support best hai
- **Flask**: Lightweight, microservices architecture support
- **MongoDB**: Flexible schema, JSON-like documents, scaling easy
- **Whisper**: Best open-source speech recognition model
- **FFmpeg**: Industry standard video processing tool
- **JWT**: Stateless authentication, scalable

---

## 📁 Project Structure (File System Ki Tafseel)

```
C:\Users\Cv\Desktop\fypdec\FYP\
│
├── 📂 src/                          # Frontend React Application
│   ├── 📂 components/               # Reusable UI Components
│   │   ├── AuthCallback.tsx         # OAuth callback handler
│   │   ├── Navbar.tsx               # Navigation bar
│   │   └── ... (other components)
│   │
│   ├── 📂 pages/                    # Page Components (Routes)
│   │   ├── Home.tsx                 # Landing page
│   │   ├── Login.tsx                # User login page
│   │   ├── Signup.tsx               # User registration
│   │   ├── Editor.tsx               # Video editor page
│   │   ├── Profile.tsx              # User profile
│   │   ├── AdminLogin.tsx           # Admin login page
│   │   ├── AdminDashboard.tsx       # Admin dashboard with charts
│   │   ├── AdminUsers.tsx           # User management
│   │   └── AdminVideos.tsx          # Video management
│   │
│   ├── 📂 contexts/                 # Global State Management
│   │   └── AuthContext.tsx          # User authentication state
│   │
│   ├── 📂 services/                 # API Service Layer
│   │   ├── api.ts                   # API configuration
│   │   └── videoService.ts          # Video-related API calls
│   │
│   ├── App.tsx                      # Main application component
│   ├── main.tsx                     # Entry point (renders App)
│   └── index.css                    # Global styles + Tailwind
│
├── 📂 backend/                      # Backend Flask Server
│   ├── app.py                       # 🔥 MAIN SERVER FILE
│   │                                # Is file mein sab routes define hain
│   │                                # Server yahan se start hota hai
│   │
│   ├── 📂 models/                   # Database Models (Schema)
│   │   ├── user.py                  # User model (registration, profile)
│   │   ├── video.py                 # Video model (upload info, status)
│   │   ├── support_ticket.py        # Support ticket model
│   │   └── admin.py                 # Admin user model
│   │
│   ├── 📂 services/                 # Business Logic Layer
│   │   ├── auth_service.py          # Authentication logic
│   │   │   ├── register_user()      # New user registration
│   │   │   ├── authenticate_user()  # Login verification
│   │   │   ├── send_verification()  # Email verification
│   │   │   └── verify_email()       # Email confirmation
│   │   │
│   │   ├── video_service.py         # Video processing logic
│   │   │   ├── process_video()      # Main video processing
│   │   │   ├── transcribe_audio()   # Whisper AI for subtitles
│   │   │   ├── enhance_audio()      # Noise reduction
│   │   │   └── generate_thumbnail() # AI thumbnail creation
│   │   │
│   │   ├── support_service.py       # Support ticket logic
│   │   │   ├── create_ticket()
│   │   │   ├── get_user_tickets()
│   │   │   └── resolve_ticket()
│   │   │
│   │   └── admin_service.py         # Admin operations
│   │       ├── get_dashboard_stats()
│   │       ├── get_all_users()
│   │       ├── get_all_videos()
│   │       └── delete_user()
│   │
│   ├── 📂 uploads/                  # Uploaded Videos & Generated Files
│   │   ├── video1.mp4
│   │   ├── video1_en.srt           # English subtitles
│   │   ├── video1_ur.srt           # Urdu subtitles
│   │   └── ... (thumbnails, processed videos)
│   │
│   ├── utils.py                     # Utility functions
│   ├── video_processor.py           # Advanced video processing
│   ├── requirements.txt             # Python dependencies list
│   └── init_db.py                   # Database initialization
│
├── 📂 docs/                         # Documentation
│   ├── ai-thumbnail-generation.md
│   ├── urdu-subtitles-enhanced.md
│   └── ... (technical docs)
│
├── package.json                     # Frontend dependencies
├── tsconfig.json                    # TypeScript configuration
├── vite.config.ts                   # Vite build configuration
├── tailwind.config.js               # Tailwind CSS settings
└── README.md                        # 📄 Yeh file!
```

---

## 🔄 Application Flow (Application Kaise Kaam Karta Hai)

### 1️⃣ User Registration Flow:

```
1. User opens browser → http://localhost:5173/signup
   ↓
2. React App.tsx renders Signup.tsx component
   ↓
3. User fills form (name, email, password)
   ↓
4. User clicks "Sign Up" button
   ↓
5. Signup.tsx calls → axios.post('http://localhost:5001/api/auth/register')
   ↓
6. Request backend ko jaata hai → app.py line 200 (@app.route('/api/auth/register'))
   ↓
7. app.py calls → auth_service.register_user(name, email, password)
   ↓
8. auth_service.py mein:
   - Password ko bcrypt se hash karta hai
   - User object create karta hai (models/user.py)
   - MongoDB mein save karta hai (db.users collection)
   - Verification email bhejta hai (SMTP server use karke)
   ↓
9. Response wapas frontend ko jaata hai
   ↓
10. Signup.tsx success message show karta hai
    ↓
11. User ko Login page par redirect karta hai
```

**Code Example (Signup.tsx):**
```typescript
const handleSubmit = async (e: React.FormEvent) => {
  e.preventDefault();
  try {
    // Backend API call
    const response = await axios.post('http://localhost:5001/api/auth/register', {
      name, email, password
    });
    
    // Success toast
    toast.success('Registration successful!');
    
    // Redirect to login
    navigate('/login');
  } catch (error) {
    toast.error('Registration failed');
  }
};
```

**Backend Code (app.py):**
```python
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    
    # Service layer ko call karo
    result = auth_service.register_user(name, email, password)
    
    if result['success']:
        return jsonify({'message': 'User registered successfully'}), 201
    else:
        return jsonify({'error': result['error']}), 400
```

---

### 2️⃣ Video Upload & Processing Flow:

```
1. User login karke → /editor page par jaata hai
   ↓
2. Editor.tsx component load hota hai
   ↓
3. User "Upload Video" button click karta hai
   ↓
4. File input dialog open hota hai
   ↓
5. User video file select karta hai (e.g., video.mp4)
   ↓
6. handleVideoUpload() function call hota hai
   ↓
7. FormData object create hota hai:
   - video: File object
   - userId: Current user ID (localStorage se)
   - language: Selected language (e.g., 'en', 'ur')
   ↓
8. axios.post('/api/videos/upload', formData) call hota hai
   ↓
9. Backend receives request → app.py line 450 (@app.route('/api/videos/upload'))
   ↓
10. app.py:
    - File ko secure_filename() se sanitize karta hai
    - /backend/uploads/ folder mein save karta hai
    - Video document create karke MongoDB mein save karta hai
    - video_service.process_video() ko background task mein call karta hai
    ↓
11. video_service.py mein process_video() function:
    
    Step 1: Video metadata extract karta hai (FFmpeg use karke)
    ↓
    Step 2: Audio extract karta hai
    ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav
    ↓
    Step 3: Noise reduction (noisereduce library)
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sample_rate)
    ↓
    Step 4: Whisper AI se transcription (subtitles)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language='ur')
    ↓
    Step 5: SRT file generate karta hai (timestamps ke sath)
    1
    00:00:00,000 --> 00:00:03,500
    Assalamu Alaikum
    ↓
    Step 6: AI se thumbnail generate karta hai
    - Video ke key frames extract karta hai (OpenCV)
    - Best frame select karta hai
    - Thumbnail save karta hai
    ↓
    Step 7: Database update karta hai
    - status: 'completed'
    - subtitle_path: 'uploads/video_ur.srt'
    - thumbnail_path: 'uploads/video_thumb.jpg'
    ↓
12. Frontend polling karta hai (har 5 seconds):
    axios.get('/api/videos/' + videoId)
    ↓
13. Jab status 'completed' ho jaaye:
    - Editor mein video show hota hai
    - Subtitles available ho jaate hain
    - Download button enable ho jaata hai
```

**Code Example (Editor.tsx - Upload):**
```typescript
const handleVideoUpload = async (file: File) => {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('userId', user?.id);
  formData.append('language', selectedLanguage);

  try {
    // Upload video
    const response = await axios.post(
      'http://localhost:5001/api/videos/upload', 
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          const percent = (progressEvent.loaded / progressEvent.total) * 100;
          setUploadProgress(percent);
        }
      }
    );

    const videoId = response.data.video_id;
    
    // Start polling for status
    const interval = setInterval(async () => {
      const statusRes = await axios.get(`http://localhost:5001/api/videos/${videoId}`);
      
      if (statusRes.data.status === 'completed') {
        clearInterval(interval);
        toast.success('Video processing completed!');
        loadVideo(videoId);
      } else if (statusRes.data.status === 'failed') {
        clearInterval(interval);
        toast.error('Video processing failed');
      }
    }, 5000); // Har 5 seconds check karo

  } catch (error) {
    toast.error('Upload failed');
  }
};
```

**Backend Code (video_service.py - Processing):**
```python
def process_video(video_id, file_path, language='en'):
    try:
        # 1. Video se audio extract karo
        audio_path = extract_audio(file_path)
        
        # 2. Noise reduction
        enhanced_audio = enhance_audio(audio_path)
        
        # 3. Whisper AI se transcription
        model = whisper.load_model("base")
        result = model.transcribe(enhanced_audio, language=language)
        
        # 4. SRT file generate karo
        srt_path = generate_srt_file(result['segments'], video_id, language)
        
        # 5. Thumbnail generate karo
        thumbnail_path = generate_thumbnail(file_path, video_id)
        
        # 6. Database update karo
        db.videos.update_one(
            {'_id': ObjectId(video_id)},
            {
                '$set': {
                    'status': 'completed',
                    'subtitle_path': srt_path,
                    'thumbnail_path': thumbnail_path,
                    'processed_at': datetime.now()
                }
            }
        )
        
        logger.info(f"Video {video_id} processed successfully")
        
    except Exception as e:
        # Error ho toh status failed mark karo
        db.videos.update_one(
            {'_id': ObjectId(video_id)},
            {'$set': {'status': 'failed', 'error_message': str(e)}}
        )
        logger.error(f"Processing failed: {str(e)}")
```

---

### 3️⃣ Admin Dashboard Flow:

```
1. Admin browser mein jaata hai → http://localhost:5173/login
   ↓
2. Login page par "Admin Access" section mein "Admin Login →" button click karta hai
   ↓
3. Navigate to → http://localhost:5173/admin/login
   ↓
4. AdminLogin.tsx component load hota hai
   ↓
5. Admin credentials enter karta hai:
   - Email: admin@snipx.com
   - Password: admin123
   ↓
6. handleLogin() function call hota hai
   ↓
7. axios.post('/api/admin/login', {email, password})
   ↓
8. Backend → app.py line 880 (@app.route('/api/admin/login'))
   ↓
9. admin_service.authenticate_admin(email, password) call hota hai
   ↓
10. admin_service.py mein:
    - Email se admin find karta hai (db.admins collection)
    - Password hash compare karta hai (bcrypt.checkpw())
    - JWT token generate karta hai
    - last_login update karta hai
    ↓
11. Response mein JWT token return hota hai
    ↓
12. AdminLogin.tsx:
    - Token ko localStorage mein save karta hai (key: 'admin_token')
    - Admin info save karta hai (key: 'admin_info')
    - Navigate to /admin/dashboard
    ↓
13. AdminDashboard.tsx component load hota hai
    ↓
14. useEffect() hook run hota hai
    ↓
15. fetchDashboardStats() function call hota hai
    ↓
16. Multiple API calls parallel mein:
    
    API 1: axios.get('/api/admin/dashboard/stats')
    → Returns: {total_users, total_videos, active_users, storage_used, etc.}
    
    API 2: axios.get('/api/admin/analytics/user-growth?days=30')
    → Returns: [{date: '2025-12-01', count: 5}, {date: '2025-12-02', count: 8}, ...]
    
    API 3: axios.get('/api/admin/analytics/video-trends?days=30')
    → Returns: [{date: '2025-12-01', count: 3}, {date: '2025-12-02', count: 7}, ...]
    
    API 4: axios.get('/api/admin/analytics/video-status')
    → Returns: {completed: 45, processing: 2, failed: 3}
    
    API 5: axios.get('/api/admin/activity?limit=10')
    → Returns: [{type: 'user_registered', email: 'user@example.com', timestamp: ...}, ...]
    ↓
17. Data receive hone ke baad state update hota hai
    ↓
18. Chart.js components re-render hote hain:
    - User Growth Chart (Line chart)
    - Video Upload Trends Chart (Line chart)
    - Video Status Distribution (Doughnut chart)
    ↓
19. Dashboard display hota hai with all stats and charts
```

**Code Example (AdminDashboard.tsx):**
```typescript
const fetchDashboardStats = async () => {
  try {
    const token = localStorage.getItem('admin_token');
    const headers = { Authorization: `Bearer ${token}` };

    // Parallel API calls
    const [statsRes, userGrowthRes, videoTrendsRes, statusRes, activityRes] = 
      await Promise.all([
        axios.get(`${API_URL}/api/admin/dashboard/stats`, { headers }),
        axios.get(`${API_URL}/api/admin/analytics/user-growth?days=30`, { headers }),
        axios.get(`${API_URL}/api/admin/analytics/video-trends?days=30`, { headers }),
        axios.get(`${API_URL}/api/admin/analytics/video-status`, { headers }),
        axios.get(`${API_URL}/api/admin/activity?limit=10`, { headers })
      ]);

    // State update
    setStats(statsRes.data);
    setUserGrowthData(userGrowthRes.data);
    setVideoTrendsData(videoTrendsRes.data);
    // ... more updates

  } catch (error) {
    if (error.response?.status === 401) {
      navigate('/admin/login'); // Unauthorized → redirect
    }
  }
};
```

**Backend Code (admin_service.py - Dashboard Stats):**
```python
def get_dashboard_stats(self):
    """
    Dashboard ke liye sare stats return karta hai
    """
    try:
        # Total users count
        total_users = self.db.users.count_documents({})
        
        # Active users (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        active_users = self.db.users.count_documents({
            'last_login': {'$gte': thirty_days_ago}
        })
        
        # Total videos
        total_videos = self.db.videos.count_documents({})
        
        # Videos by status
        completed_videos = self.db.videos.count_documents({'status': 'completed'})
        processing_videos = self.db.videos.count_documents({'status': 'processing'})
        failed_videos = self.db.videos.count_documents({'status': 'failed'})
        
        # Storage calculation (sum of all video file sizes)
        pipeline = [
            {'$group': {'_id': None, 'total': {'$sum': '$file_size'}}}
        ]
        storage_result = list(self.db.videos.aggregate(pipeline))
        storage_used = storage_result[0]['total'] if storage_result else 0
        
        # Support tickets stats
        total_tickets = self.db.support_tickets.count_documents({})
        open_tickets = self.db.support_tickets.count_documents({'status': 'open'})
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'total_videos': total_videos,
            'completed_videos': completed_videos,
            'processing_videos': processing_videos,
            'failed_videos': failed_videos,
            'storage_used_mb': round(storage_used / (1024 * 1024), 2),
            'total_tickets': total_tickets,
            'open_tickets': open_tickets
        }
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {str(e)}")
        return {}
```

---

## 🔐 Authentication System (Login/Registration Kaise Kaam Karta Hai)

### JWT Token Authentication:

```
1. User login karta hai
   ↓
2. Backend email aur password verify karta hai
   ↓
3. JWT token generate hota hai:
   token = jwt.encode(
       {
           'user_id': str(user_id),
           'email': user.email,
           'exp': datetime.now() + timedelta(hours=24)
       },
       SECRET_KEY,
       algorithm='HS256'
   )
   ↓
4. Token frontend ko send hota hai
   ↓
5. Frontend token ko localStorage mein save karta hai
   ↓
6. Har API request mein token bhejta hai:
   headers: { Authorization: 'Bearer <token>' }
   ↓
7. Backend har request par token verify karta hai:
   - Token decode karta hai
   - Expiry check karta hai
   - User ID extract karta hai
   ↓
8. Agar token valid hai → Request process hota hai
   Agar token invalid/expired → 401 Unauthorized response
```

**Middleware (app.py):**
```python
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        
        try:
            # Bearer token se actual token extract karo
            token = token.split(' ')[1]
            
            # Token decode karo
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            
            # User ID request mein attach karo
            request.user_id = payload['user_id']
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(*args, **kwargs)
    
    return decorated

# Usage example
@app.route('/api/videos/my-videos', methods=['GET'])
@require_auth  # This route requires authentication
def get_my_videos():
    user_id = request.user_id  # Available from middleware
    videos = db.videos.find({'user_id': user_id})
    return jsonify({'videos': list(videos)})
```

---

## 📊 Database Schema (MongoDB Collections)

### 1. Users Collection:
```javascript
{
  _id: ObjectId("67abc..."),
  name: "Ahmed Khan",
  email: "ahmed@example.com",
  password_hash: "$2b$12$xyz...",  // bcrypt hash
  is_verified: true,
  verification_token: null,
  created_at: ISODate("2025-12-01T10:30:00Z"),
  last_login: ISODate("2025-12-25T15:45:00Z"),
  profile_picture: "uploads/profile_123.jpg",
  total_videos: 15,
  storage_used: 524288000  // bytes
}
```

### 2. Videos Collection:
```javascript
{
  _id: ObjectId("67def..."),
  user_id: ObjectId("67abc..."),
  user_email: "ahmed@example.com",
  filename: "my_video.mp4",
  original_filename: "My Amazing Video.mp4",
  file_path: "uploads/my_video_1735123456.mp4",
  file_size: 15728640,  // 15 MB in bytes
  duration: 125.5,  // seconds
  status: "completed",  // processing | completed | failed
  language: "ur",  // Urdu
  subtitle_path: "uploads/my_video_ur.srt",
  thumbnail_path: "uploads/my_video_thumb.jpg",
  created_at: ISODate("2025-12-25T12:00:00Z"),
  processed_at: ISODate("2025-12-25T12:05:30Z"),
  error_message: null,
  processing_logs: [
    { timestamp: "2025-12-25T12:01:00Z", step: "audio_extraction", status: "success" },
    { timestamp: "2025-12-25T12:02:30Z", step: "transcription", status: "success" }
  ]
}
```

### 3. Admins Collection:
```javascript
{
  _id: ObjectId("67ghi..."),
  email: "admin@snipx.com",
  password_hash: "$2b$12$abc...",
  name: "Admin User",
  role: "super_admin",  // super_admin | admin | moderator
  permissions: [
    "view_users",
    "edit_users",
    "delete_users",
    "view_videos",
    "delete_videos",
    "view_analytics",
    "manage_admins",
    "system_settings"
  ],
  created_at: ISODate("2025-12-20T08:00:00Z"),
  last_login: ISODate("2025-12-25T16:30:00Z")
}
```

### 4. Support Tickets Collection:
```javascript
{
  _id: ObjectId("67jkl..."),
  user_id: ObjectId("67abc..."),
  user_email: "ahmed@example.com",
  user_name: "Ahmed Khan",
  subject: "Video processing failed",
  message: "Meri video upload toh ho gayi lekin processing mein error aa raha hai",
  status: "open",  // open | in_progress | resolved | closed
  priority: "high",  // low | medium | high | urgent
  category: "technical",  // technical | billing | general
  created_at: ISODate("2025-12-25T14:20:00Z"),
  updated_at: ISODate("2025-12-25T14:20:00Z"),
  resolved_at: null,
  admin_reply: null,
  admin_id: null
}
```

---

## 🚀 Installation & Setup (Installation Kaise Karein)

### Prerequisites (Pehle Yeh Install Karein):

1. **Node.js** (v18 ya higher)
   - Download: https://nodejs.org/
   - Verify: `node --version`

2. **Python** (v3.12)
   - Download: https://www.python.org/downloads/
   - Verify: `python --version`

3. **MongoDB** (Community Edition)
   - Download: https://www.mongodb.com/try/download/community
   - Start service: `net start MongoDB`

4. **FFmpeg** (Video processing ke liye)
   - Download: https://ffmpeg.org/download.html
   - Add to PATH environment variable
   - Verify: `ffmpeg -version`

### Step 1: Project Clone/Download
```bash
cd C:\Users\Cv\Desktop\fypdec\FYP
```

### Step 2: Frontend Setup
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```
Server chalu ho jayega: **http://localhost:5173**

### Step 3: Backend Setup
```bash
# Backend folder mein jao
cd backend

# Virtual environment create karo (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Dependencies install karo
pip install -r requirements.txt

# Database initialize karo
python init_db.py

# Server start karo
python app.py
```
Server chalu ho jayega: **http://localhost:5001**

### Step 4: Environment Variables (Optional)
Create `.env` file in backend folder:
```env
MONGODB_URI=mongodb://localhost:27017/snipx
SECRET_KEY=your_secret_key_here
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### Step 5: Create First Admin User
```bash
cd backend
python

>>> from app import db, admin_service
>>> admin_service.create_admin('admin@snipx.com', 'admin123', 'Admin User', 'super_admin')
>>> exit()
```

---

## 🎯 Important Functions (Ahem Functions Ki List)

### Frontend Functions:

#### 1. **AuthContext (src/contexts/AuthContext.tsx)**
```typescript
// User login function
const login = async (email: string, password: string) => {
  const response = await axios.post('/api/auth/login', { email, password });
  const { token, user } = response.data;
  
  // Token save karo
  localStorage.setItem('token', token);
  localStorage.setItem('user', JSON.stringify(user));
  
  // State update karo
  setUser(user);
  setIsAuthenticated(true);
};

// User logout function
const logout = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  setUser(null);
  setIsAuthenticated(false);
  navigate('/login');
};

// Demo login (testing ke liye)
const loginAsDemo = async () => {
  await login('demo@snipx.com', 'demo1234');
};
```

#### 2. **Video Upload (src/pages/Editor.tsx)**
```typescript
const handleVideoUpload = async (file: File) => {
  // FormData create karo
  const formData = new FormData();
  formData.append('video', file);
  formData.append('userId', user?.id);
  formData.append('language', selectedLanguage);
  
  // Progress bar ke sath upload karo
  const response = await axios.post('/api/videos/upload', formData, {
    onUploadProgress: (e) => {
      setProgress((e.loaded / e.total) * 100);
    }
  });
  
  // Video processing status check karo
  pollVideoStatus(response.data.video_id);
};

const pollVideoStatus = (videoId: string) => {
  const interval = setInterval(async () => {
    const res = await axios.get(`/api/videos/${videoId}`);
    
    if (res.data.status === 'completed') {
      clearInterval(interval);
      // Video ready hai
      loadVideo(videoId);
    }
  }, 5000);
};
```

#### 3. **Admin Dashboard Stats (src/pages/AdminDashboard.tsx)**
```typescript
const fetchDashboardStats = async () => {
  const token = localStorage.getItem('admin_token');
  const headers = { Authorization: `Bearer ${token}` };
  
  // Multiple API calls parallel mein
  const [stats, userGrowth, videoTrends] = await Promise.all([
    axios.get('/api/admin/dashboard/stats', { headers }),
    axios.get('/api/admin/analytics/user-growth?days=30', { headers }),
    axios.get('/api/admin/analytics/video-trends?days=30', { headers })
  ]);
  
  // State update
  setStats(stats.data);
  setUserGrowthData(userGrowth.data);
  setVideoTrendsData(videoTrends.data);
  
  // Chart.js ko data pass karo
  createCharts();
};
```

### Backend Functions:

#### 1. **User Registration (backend/services/auth_service.py)**
```python
def register_user(self, name, email, password):
    """
    Naya user create karta hai aur verification email bhejta hai
    """
    # Check if email already exists
    if self.db.users.find_one({'email': email}):
        return {'success': False, 'error': 'Email already exists'}
    
    # Password hash karo (bcrypt use karke)
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Verification token generate karo
    verification_token = secrets.token_urlsafe(32)
    
    # User object create karo
    user = User(
        name=name,
        email=email,
        password_hash=password_hash.decode('utf-8'),
        verification_token=verification_token,
        is_verified=False,
        created_at=datetime.now()
    )
    
    # Database mein save karo
    result = self.db.users.insert_one(user.to_dict())
    user_id = str(result.inserted_id)
    
    # Verification email bhejo
    self.send_verification_email(email, verification_token)
    
    return {
        'success': True,
        'user_id': user_id,
        'message': 'User registered. Please check email for verification.'
    }
```

#### 2. **Video Processing (backend/services/video_service.py)**
```python
def process_video(self, video_id, file_path, language='en'):
    """
    Main video processing function - subtitles aur thumbnail generate karta hai
    """
    try:
        logger.info(f"Starting processing for video {video_id}")
        
        # Step 1: Audio extract karo
        audio_path = self.extract_audio(file_path)
        self.update_processing_log(video_id, 'audio_extraction', 'success')
        
        # Step 2: Noise reduction
        enhanced_audio = self.enhance_audio(audio_path)
        self.update_processing_log(video_id, 'noise_reduction', 'success')
        
        # Step 3: Whisper AI se transcription
        subtitles = self.transcribe_audio(enhanced_audio, language)
        self.update_processing_log(video_id, 'transcription', 'success')
        
        # Step 4: SRT file generate karo
        srt_path = self.generate_srt(subtitles, video_id, language)
        self.update_processing_log(video_id, 'srt_generation', 'success')
        
        # Step 5: AI thumbnail generate karo
        thumbnail_path = self.generate_thumbnail(file_path, video_id)
        self.update_processing_log(video_id, 'thumbnail_generation', 'success')
        
        # Step 6: Database update - status 'completed' mark karo
        self.db.videos.update_one(
            {'_id': ObjectId(video_id)},
            {
                '$set': {
                    'status': 'completed',
                    'subtitle_path': srt_path,
                    'thumbnail_path': thumbnail_path,
                    'processed_at': datetime.now()
                }
            }
        )
        
        logger.info(f"Video {video_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Processing failed for {video_id}: {str(e)}")
        
        # Error hui toh status 'failed' mark karo
        self.db.videos.update_one(
            {'_id': ObjectId(video_id)},
            {
                '$set': {
                    'status': 'failed',
                    'error_message': str(e)
                }
            }
        )

def transcribe_audio(self, audio_path, language='en'):
    """
    OpenAI Whisper AI se audio ko text mein convert karta hai
    """
    # Whisper model load karo (base model - balance between speed and accuracy)
    model = whisper.load_model("base")
    
    # Transcription karo
    result = model.transcribe(
        audio_path,
        language=language,  # ur (Urdu), en (English), ar (Arabic), etc.
        fp16=False  # CPU ke liye
    )
    
    # Segments return karo (har segment mein text aur timestamps)
    return result['segments']
    # Example output:
    # [
    #   {'start': 0.0, 'end': 3.5, 'text': 'Assalamu Alaikum'},
    #   {'start': 3.5, 'end': 7.2, 'text': 'Aaj hum video processing ke baare mein baat karenge'}
    # ]

def generate_srt(self, segments, video_id, language):
    """
    Whisper segments se SRT subtitle file generate karta hai
    """
    srt_content = ""
    
    for i, segment in enumerate(segments, 1):
        # Timestamp format: HH:MM:SS,mmm
        start_time = self.format_timestamp(segment['start'])
        end_time = self.format_timestamp(segment['end'])
        text = segment['text'].strip()
        
        # SRT format:
        # 1
        # 00:00:00,000 --> 00:00:03,500
        # Assalamu Alaikum
        #
        srt_content += f"{i}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{text}\n\n"
    
    # File save karo
    srt_filename = f"{video_id}_{language}.srt"
    srt_path = os.path.join('uploads', srt_filename)
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    return srt_path
```

---

## 📡 API Endpoints (Complete List)

### Authentication APIs:
```
POST   /api/auth/register              # New user registration
POST   /api/auth/login                 # User login (returns JWT token)
POST   /api/auth/verify-email          # Email verification
GET    /api/auth/google/login          # Google OAuth login
GET    /api/auth/google/callback       # OAuth callback
POST   /api/auth/logout                # User logout
```

### Video APIs:
```
POST   /api/videos/upload              # Upload new video
GET    /api/videos/:id                 # Get video details
GET    /api/videos/my-videos           # Get current user's videos
DELETE /api/videos/:id                 # Delete video
GET    /api/videos/:id/subtitles       # Download subtitles
GET    /api/videos/:id/thumbnail       # Get thumbnail
POST   /api/videos/:id/process         # Re-process video
```

### User APIs:
```
GET    /api/users/profile              # Get current user profile
PUT    /api/users/profile              # Update profile
POST   /api/users/change-password      # Change password
GET    /api/users/:id                  # Get user by ID (admin only)
```

### Support APIs:
```
POST   /api/support/tickets            # Create new ticket
GET    /api/support/tickets            # Get user's tickets
GET    /api/support/tickets/:id        # Get ticket details
PUT    /api/support/tickets/:id        # Update ticket
```

### Admin APIs:
```
POST   /api/admin/login                      # Admin login
GET    /api/admin/dashboard/stats            # Dashboard statistics
GET    /api/admin/analytics/user-growth      # User growth chart data
GET    /api/admin/analytics/video-trends     # Video upload trends
GET    /api/admin/analytics/video-status     # Video status distribution
GET    /api/admin/analytics/content          # Content analytics
GET    /api/admin/users                      # Get all users (pagination)
GET    /api/admin/users/:id                  # Get user details
PUT    /api/admin/users/:id                  # Update user
DELETE /api/admin/users/:id                  # Delete user
POST   /api/admin/users/:id/toggle-status    # Activate/deactivate user
GET    /api/admin/videos                     # Get all videos (filters)
GET    /api/admin/videos/:id/logs            # Get video processing logs
DELETE /api/admin/videos/:id                 # Delete video
GET    /api/admin/activity                   # Recent system activity
```

---

## 🔧 Important Modules & Their Purpose

### Frontend Modules:

1. **React Router** (`react-router-dom`)
   - Purpose: Page navigation aur routing
   - Usage: `<Route path="/editor" element={<Editor />} />`

2. **Axios** (`axios`)
   - Purpose: HTTP requests (API calls)
   - Usage: `axios.post('/api/videos/upload', formData)`

3. **Chart.js** (`chart.js`, `react-chartjs-2`)
   - Purpose: Data visualization (graphs aur charts)
   - Usage: Admin dashboard mein user growth, video trends charts

4. **Tailwind CSS** (`tailwindcss`)
   - Purpose: Utility-first CSS framework
   - Usage: `className="bg-blue-500 hover:bg-blue-700 px-4 py-2"`

5. **React Hot Toast** (`react-hot-toast`)
   - Purpose: Toast notifications (success/error messages)
   - Usage: `toast.success('Video uploaded!')`

### Backend Modules:

1. **Flask** (`flask`)
   - Purpose: Web framework (routes aur API endpoints)
   - Usage: `@app.route('/api/videos/upload', methods=['POST'])`

2. **PyMongo** (`pymongo`)
   - Purpose: MongoDB database connection aur queries
   - Usage: `db.users.find_one({'email': email})`

3. **OpenAI Whisper** (`openai-whisper`)
   - Purpose: Speech-to-text (automatic subtitles generation)
   - Usage: `model.transcribe(audio_path, language='ur')`

4. **Noisereduce** (`noisereduce`)
   - Purpose: Audio noise removal
   - Usage: `nr.reduce_noise(y=audio_data, sr=sample_rate)`

5. **FFmpeg-Python** (`ffmpeg-python`)
   - Purpose: Video/audio processing
   - Usage: Extract audio, merge subtitles, compress videos

6. **OpenCV** (`opencv-python`)
   - Purpose: Computer vision (thumbnail generation)
   - Usage: Extract key frames from video

7. **bcrypt** (`bcrypt`)
   - Purpose: Password hashing (security)
   - Usage: `bcrypt.hashpw(password, bcrypt.gensalt())`

8. **JWT** (`pyjwt`)
   - Purpose: Token-based authentication
   - Usage: `jwt.encode({'user_id': id}, SECRET_KEY)`

9. **Werkzeug** (`werkzeug`)
   - Purpose: File upload handling, security utilities
   - Usage: `secure_filename(file.filename)`

---

## 📝 Common Issues & Solutions

### Issue 1: MongoDB Connection Failed
```
Error: MongoServerError: connect ECONNREFUSED
```
**Solution:**
```bash
# MongoDB service start karo
net start MongoDB

# Ya phir manually start karo
"C:\Program Files\MongoDB\Server\7.0\bin\mongod.exe" --dbpath "C:\data\db"
```

### Issue 2: FFmpeg Not Found
```
Error: ffmpeg: command not found
```
**Solution:**
1. FFmpeg download karo: https://ffmpeg.org/download.html
2. Extract karo (e.g., `C:\ffmpeg`)
3. Environment variables mein add karo:
   - Path: `C:\ffmpeg\bin`
4. CMD restart karo aur verify karo: `ffmpeg -version`

### Issue 3: Whisper Model Download Issue
```
Error: Failed to download Whisper model
```
**Solution:**
```python
# Pehli baar manually download karo
import whisper
model = whisper.load_model("base", download_root="./models")
```

### Issue 4: Port Already in Use
```
Error: Address already in use: 5001
```
**Solution:**
```bash
# Port par running process find karo
netstat -ano | findstr :5001

# Process ko kill karo (PID se)
taskkill /PID <process_id> /F
```

### Issue 5: CORS Error in Browser
```
Error: Access to XMLHttpRequest blocked by CORS policy
```
**Solution:**
```python
# app.py mein CORS properly configure karo
from flask_cors import CORS
CORS(app, origins=["http://localhost:5173"])
```

---

## 🚀 Production Deployment

### Frontend (Vite Build):
```bash
# Production build
npm run build

# dist/ folder generate hoga
# Is folder ko Netlify/Vercel par deploy karo
```

### Backend (Flask Production):
```bash
# Gunicorn install karo (production WSGI server)
pip install gunicorn

# Server start karo
gunicorn -w 4 -b 0.0.0.0:5001 app:app

# -w 4: 4 worker processes
# -b 0.0.0.0:5001: Bind to all interfaces on port 5001
```

### MongoDB Atlas (Cloud Database):
```python
# .env file mein MongoDB Atlas URI add karo
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/snipx?retryWrites=true&w=majority

# app.py mein use karo
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv('MONGODB_URI')
client = MongoClient(MONGO_URI)
```

---

## 👨‍💻 Developer Notes

### Code Organization:
- **Separation of Concerns**: Models, Services, aur Routes alag files mein
- **Service Layer Pattern**: Business logic services folder mein
- **Context API**: Global state management (authentication)
- **Component Reusability**: Common components reuse karo

### Best Practices:
1. **Error Handling**: Har function mein try-catch use karo
2. **Logging**: Important events log karo (logger.info, logger.error)
3. **Validation**: User input ko always validate karo
4. **Comments**: Complex logic ko explain karo (Roman Urdu ya English)
5. **Git Commits**: Meaningful commit messages likho

### Performance Tips:
- **Lazy Loading**: Large components ko lazy load karo
- **Caching**: Frequently accessed data cache karo (Redis)
- **Database Indexes**: Query performance ke liye indexes banao
- **Image Optimization**: Thumbnails ko compress karo
- **CDN**: Static files (videos, images) ke liye CDN use karo

---

## 📞 Contact & Support

**Developer:** Ahmed (CV)  
**Project Type:** Final Year Project (FYP)  
**University:** [Your University Name]  
**Year:** 2025

### Tech Stack Summary:
- Frontend: React + TypeScript + Vite
- Backend: Flask + Python
- Database: MongoDB
- AI/ML: OpenAI Whisper, TensorFlow
- Video Processing: FFmpeg, OpenCV

---

## 📜 License

This project is for educational purposes (Final Year Project).  
© 2025 SnipX Technologies. All rights reserved.

---

## 🙏 Acknowledgments

Special thanks to:
- OpenAI for Whisper model
- MongoDB community
- Flask & React communities
- All open-source contributors

---

**Last Updated:** December 25, 2025  
**Version:** 1.0.0  
**Status:** ✅ Fully Functional

---

## Quick Start Commands:

```bash
# Frontend start karo
npm run dev

# Backend start karo
cd backend
python app.py

# MongoDB start karo
net start MongoDB

# Admin user create karo
cd backend
python
>>> from app import admin_service
>>> admin_service.create_admin('admin@snipx.com', 'admin123', 'Admin', 'super_admin')
```

**Admin Dashboard Access:**
1. Browser mein jao: http://localhost:5173/login
2. Neeche "Admin Access" section mein "Admin Login →" click karo
3. Login karo: admin@snipx.com / admin123
4. Dashboard access ho jayega! 🎉

---

*Yeh README.md file complete guide hai SnipX platform ki. Agar koi question ho toh feel free to ask!* 😊
