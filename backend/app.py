from flask import Flask, request, jsonify, redirect, url_for, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId
from authlib.integrations.flask_client import OAuth
from datetime import datetime
from dotenv import load_dotenv
import logging
import os
from bson import ObjectId

from services.auth_service import AuthService
from services.video_service import VideoService
from services.support_service import SupportService
from services.admin_service import AdminService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

# App secret
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 500 * 1024 * 1024))

# MongoDB connection for production
def connect_mongodb():
    """Connect to MongoDB (Atlas for production, local for development)"""
    mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
    
    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=10000)
        client.server_info()
        
        # Check if it's Atlas or local
        if 'mongodb+srv' in mongodb_uri or 'mongodb.net' in mongodb_uri:
            logger.info("✅ Connected to MongoDB Atlas (production)")
        else:
            logger.info("✅ Connected to local MongoDB (development)")
        
        return client
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {str(e)}")
        logger.error(f"❌ URI: {mongodb_uri[:30]}...")
        raise Exception(f"Could not connect to MongoDB: {str(e)}")

try:
    client = connect_mongodb()
    db = client.snipx
except Exception as e:
    logger.error(f"❌ All MongoDB connections failed: {str(e)}")
    raise

# Initialize services
auth_service = AuthService(db)
video_service = VideoService(db)
support_service = SupportService(db)
admin_service = AdminService(db)

# OAuth setup
oauth = OAuth(app)

oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    client_kwargs={'scope': 'openid email profile'},
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration'
)



@app.route('/api/auth/google/login')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@app.route('/api/auth/google/callback')
def google_callback():
    token = oauth.google.authorize_access_token()
    user_info = oauth.google.parse_id_token(token, nonce=None)

    user = db.users.find_one({'email': user_info['email']})
    if not user:
        user_id = str(db.users.insert_one({
            'email': user_info['email'],
            'first_name': user_info.get('given_name'),
            'last_name': user_info.get('family_name'),
            'oauth_id': user_info['sub'],
            'provider': 'google',
            'created_at': datetime.utcnow()
        }).inserted_id)
    else:
        user_id = str(user['_id'])

    jwt_token = auth_service.generate_token(user_id)
    return redirect(f"http://localhost:5173/auth/callback?token={jwt_token}")



def require_auth(f):
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        print(f"[AUTH] Auth header: {auth_header[:50] if auth_header else 'None'}...")
        if not auth_header:
            return jsonify({'error': 'No authorization header'}), 401

        try:
            token = auth_header.split(' ')[1]
            user_id = auth_service.verify_token(token)
            print(f"[AUTH] User ID from token: {user_id}")
            return f(user_id, *args, **kwargs)
        except Exception as e:
            print(f"[AUTH] Token verification failed: {e}")
            return jsonify({'error': str(e)}), 401

    decorated.__name__ = f.__name__
    return decorated

@app.route('/api/test-db', methods=['GET'])
def test_db():
    try:
        test_result = db.users.find_one()
        if test_result and '_id' in test_result:
            test_result['_id'] = str(test_result['_id'])

        return jsonify({
            "status": "success",
            "message": "MongoDB is connected",
            "sample_user": test_result
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "MongoDB connection failed",
            "error": str(e)
        }), 500

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        if not all(field in data for field in ['email', 'password']):
            return jsonify({'error': 'Missing required fields'}), 400

        user_id = auth_service.register_user(
            email=data['email'],
            password=data['password'],
            first_name=data.get('firstName'),
            last_name=data.get('lastName')
        )
        return jsonify({'message': 'User registered successfully', 'user_id': str(user_id)}), 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception("Signup error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or not all(field in data for field in ['email', 'password']):
            return jsonify({'message': 'Missing credentials'}), 400

        token, user = auth_service.login_user(data['email'], data['password'])
        return jsonify({'token': token, 'user': user}), 200
    except ValueError as e:
        return jsonify({'message': str(e)}), 401
    except Exception as e:
        logger.exception("Login error")
        return jsonify({'message': str(e)}), 500

@app.route('/api/auth/demo', methods=['POST'])
def demo_login():
    try:
        token, user = auth_service.create_demo_user()
        return jsonify({'token': token, 'user': user}), 200
    except Exception as e:
        logger.exception("Demo login error")
        return jsonify({'message': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        conversation_history = data.get('history', [])
        
        # Simple fallback response for now
        response = "I'm a basic chatbot. For detailed support, please use the support ticket system in the Help section."
        
        return jsonify({
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.exception("Chat error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/support/tickets', methods=['POST'])
@require_auth
def create_support_ticket(user_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['name', 'email', 'subject', 'description', 'priority', 'type']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        ticket_id = support_service.create_ticket(user_id, data)
        
        return jsonify({
            'message': 'Support ticket created successfully',
            'ticket_id': ticket_id
        }), 201
        
    except Exception as e:
        logger.exception("Support ticket creation error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/support/tickets', methods=['GET'])
@require_auth
def get_support_tickets(user_id):
    try:
        # Get tickets for the authenticated user
        tickets = support_service.get_user_tickets(user_id)
        return jsonify(tickets), 200
    except Exception as e:
        logger.exception("Get support tickets error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/support/tickets/<ticket_id>', methods=['GET'])
@require_auth
def get_support_ticket(user_id, ticket_id):
    try:
        ticket = support_service.get_ticket(ticket_id)
        if not ticket:
            return jsonify({'error': 'Ticket not found'}), 404
        
        # Ensure user owns the ticket
        if str(ticket.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized access to ticket'}), 403
            
        return jsonify(ticket.to_dict()), 200
    except Exception as e:
        logger.exception("Get support ticket error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_video(user_id):
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        video_id = video_service.save_video(file, user_id)

        return jsonify({'message': 'Video uploaded successfully', 'video_id': str(video_id)}), 200
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/process', methods=['POST'])
@require_auth
def process_video(user_id, video_id):
    try:
        options = request.json.get('options', {})
        logger.info(f"Processing video {video_id} with options: {options}")
        print(f"[PROCESS] Video ID: {video_id}")
        print(f"[PROCESS] Options received: {options}")
        print(f"[PROCESS] generate_thumbnail: {options.get('generate_thumbnail', False)}")
        print(f"[PROCESS] thumbnail_text: '{options.get('thumbnail_text')}'")
        print(f"[PROCESS] thumbnail_frame_index: {options.get('thumbnail_frame_index')}")
        
        video_service.process_video(video_id, options)
        
        # Get updated video to check outputs
        video = video_service.get_video(video_id)
        if video:
            print(f"[PROCESS] Video outputs after processing: {video.outputs}")
            logger.info(f"Video outputs: {video.outputs}")
        
        return jsonify({'message': 'Processing completed successfully'}), 200
    except Exception as e:
        logger.error(f"Process error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>', methods=['GET'])
@require_auth
def get_video_status(user_id, video_id):
    try:
        logger.info(f"Getting video status for video_id: {video_id}, user_id: {user_id}")
        print(f"[GET_VIDEO] Fetching video {video_id}")
        
        video = video_service.get_video(video_id)
        if not video:
            logger.error(f"Video not found: {video_id}")
            print(f"[GET_VIDEO] Video not found: {video_id}")
            return jsonify({'error': 'Video not found'}), 404

        print(f"[GET_VIDEO] Found video: {video.filename}, status: {video.status}")

        # Convert custom Video object to dict
        if hasattr(video, 'to_dict'):
            video_dict = video.to_dict()
        elif hasattr(video, '__dict__'):
            video_dict = video.__dict__
        else:
            raise ValueError("Cannot serialize Video object")

        # Clean up any non-serializable fields (e.g., ObjectId)
        if '_id' in video_dict:
            video_dict['_id'] = str(video_dict['_id'])
        if 'user_id' in video_dict:
            video_dict['user_id'] = str(video_dict['user_id'])

        print(f"[GET_VIDEO] Returning video data: status={video_dict.get('status')}, outputs={video_dict.get('outputs')}")
        return jsonify(video_dict), 200

    except Exception as e:
        logger.error(f"Fetch video error: {str(e)}")
        print(f"[GET_VIDEO] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos', methods=['GET'])
@require_auth
def get_user_videos(user_id):
    try:
        print(f"[GET_VIDEOS] Fetching videos for user_id: {user_id}")
        videos = video_service.get_user_videos(user_id)
        print(f"[GET_VIDEOS] Found {len(videos)} videos")
        return jsonify(videos), 200
    except Exception as e:
        logger.error(f"List videos error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>', methods=['DELETE'])
@require_auth
def delete_video(user_id, video_id):
    try:
        video_service.delete_video(video_id, user_id)
        return jsonify({'message': 'Video deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Delete video error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Export/Render video with all edits (trim, text overlay, music)
@app.route('/api/videos/<video_id>/export', methods=['POST'])
@require_auth
def export_video(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json() or {}
        
        # Get edit parameters
        trim_start = data.get('trim_start', 0)  # Percentage 0-100
        trim_end = data.get('trim_end', 100)  # Percentage 0-100
        text_overlay = data.get('text_overlay', '')
        text_position = data.get('text_position', 'center')
        text_color = data.get('text_color', '#ffffff')
        text_size = data.get('text_size', 32)
        music_volume = data.get('music_volume', 50)
        video_volume = data.get('video_volume', 100)
        mute_original = data.get('mute_original', False)
        
        logger.info(f"Exporting video {video_id} with edits: trim={trim_start}-{trim_end}%, text='{text_overlay}'")
        
        # Process video with edits using video service
        export_path = video_service.export_video_with_edits(
            video_id=video_id,
            trim_start=trim_start,
            trim_end=trim_end,
            text_overlay=text_overlay,
            text_position=text_position,
            text_color=text_color,
            text_size=text_size,
            music_volume=music_volume,
            video_volume=video_volume,
            mute_original=mute_original
        )
        
        if not export_path or not os.path.exists(export_path):
            return jsonify({'error': 'Export failed'}), 500
        
        # Return the download URL
        return jsonify({
            'message': 'Video exported successfully',
            'download_url': f'/api/videos/{video_id}/download-export'
        }), 200
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Download exported video
@app.route('/api/videos/<video_id>/download-export', methods=['GET'])
@require_auth
def download_exported_video(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get the exported video path
        export_path = video.outputs.get('exported_video', video.outputs.get('processed_video', video.filepath))
        
        if not os.path.exists(export_path):
            return jsonify({'error': 'Exported video not found'}), 404
        
        # Generate filename
        base_name = os.path.splitext(video.filename)[0]
        download_name = f"{base_name}_edited.mp4"
        
        return send_file(
            export_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=download_name,
            conditional=False
        )
    except Exception as e:
        logger.error(f"Download export error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Add download endpoint for processed videos
@app.route('/api/videos/<video_id>/download', methods=['GET'])
@require_auth
def download_video(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get the processed video path
        processed_path = video.outputs.get('processed_video', video.filepath)
        
        if not os.path.exists(processed_path):
            return jsonify({'error': 'Processed video not found'}), 404
        
        # Ensure the file is properly closed before sending
        return send_file(
            processed_path,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=f"enhanced_{video.filename}",
            conditional=False
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/thumbnails/generate', methods=['POST'])
@require_auth
def generate_thumbnails(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json() or {}
        count = data.get('count', 5)
        style = data.get('style', 'auto')
        
        # Generate thumbnails
        video_service._generate_thumbnail(video)
        
        # Update video in database
        video_service.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": video.to_dict()}
        )
        
        return jsonify({
            'message': 'Thumbnails generated successfully',
            'thumbnails': video.outputs.get('thumbnails', []),
            'count': len(video.outputs.get('thumbnails', []))
        }), 200
        
    except Exception as e:
        logger.error(f"Generate thumbnails error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/audio/enhance', methods=['POST'])
@require_auth
def enhance_audio_realtime(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json() or {}
        enhancement_type = data.get('type', 'full')
        noise_reduction = data.get('noiseReduction', True)
        volume_boost = data.get('volumeBoost', 20)
        
        # Enhanced audio processing options
        options = {
            'audio_enhancement_type': enhancement_type,
            'noise_reduction': noise_reduction,
            'volume_boost': volume_boost,
            'enhance_audio': True
        }
        
        video_service._enhance_audio(video, options)
        
        # Update video in database
        video_service.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": video.to_dict()}
        )
        
        return jsonify({
            'message': 'Audio enhanced successfully',
            'enhancement_type': enhancement_type,
            'processed_audio': video.outputs.get('processed_video')
        }), 200
        
    except Exception as e:
        logger.error(f"Audio enhancement error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/status', methods=['GET'])
@require_auth
def get_processing_status(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Return detailed processing status
        status_info = {
            'status': video.status,
            'progress': {
                'upload': 100 if video.status != 'uploading' else 50,
                'processing': 100 if video.status == 'completed' else (75 if video.status == 'processing' else 0),
                'thumbnails': 100 if video.outputs.get('thumbnails') else 0,
                'audio_enhancement': 100 if video.outputs.get('processed_video') else 0,
                'subtitles': 100 if video.outputs.get('subtitles') else 0
            },
            'outputs': {
                'thumbnails_count': len(video.outputs.get('thumbnails', [])),
                'has_enhanced_audio': bool(video.outputs.get('processed_video')),
                'has_subtitles': bool(video.outputs.get('subtitles')),
                'has_summary': bool(video.outputs.get('summary'))
            },
            'metadata': video.metadata,
            'processing_time': (
                (video.process_end_time - video.process_start_time).total_seconds()
                if video.process_start_time and video.process_end_time
                else None
            )
        }
        
        return jsonify(status_info), 200
        
    except Exception as e:
        logger.error(f"Get processing status error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/subtitles', methods=['GET'])
@require_auth
def get_video_subtitles(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        subtitles_info = video.outputs.get('subtitles', {})
        if not subtitles_info:
            return jsonify([]), 200
        
        json_path = subtitles_info.get('json')
        if not json_path or not os.path.exists(json_path):
            return jsonify([]), 200
        
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            subtitle_data = json.load(f)
        
        return jsonify(subtitle_data.get('segments', [])), 200
        
    except Exception as e:
        logger.error(f"Get subtitles error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/subtitles/<language>/download', methods=['GET'])
@require_auth
def download_subtitles(user_id, video_id, language):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        format_type = request.args.get('format', 'srt')
        subtitles_info = video.outputs.get('subtitles', {})
        
        if not subtitles_info:
            return jsonify({'error': 'No subtitles found'}), 404
        
        # If it's a string (old format), try to find the file
        if isinstance(subtitles_info, str):
            subtitle_path = subtitles_info
        else:
            subtitle_path = subtitles_info.get('srt' if format_type == 'srt' else 'json')
        
        if not subtitle_path or not os.path.exists(subtitle_path):
            return jsonify({'error': 'Subtitle file not found'}), 404
        
        filename = f"{video.filename}_{language}.{format_type}"
        return send_file(
            subtitle_path,
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Download subtitles error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/subtitles/generate', methods=['POST'])
@require_auth
def generate_subtitles(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        language = data.get('language', 'en')
        style = data.get('style', 'clean')
        
        # Generate subtitles
        options = {
            'subtitle_language': language,
            'subtitle_style': style,
            'generate_subtitles': True
        }
        
        video_service._generate_subtitles(video, options)
        
        # Update video in database
        video_service.videos.update_one(
            {"_id": ObjectId(video_id)},
            {"$set": video.to_dict()}
        )
        
        return jsonify({
            'message': 'Subtitles generated successfully',
            'language': language,
            'style': style
        }), 200
        
    except Exception as e:
        logger.error(f"Generate subtitles error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/thumbnail', methods=['GET'])
def get_video_thumbnail(video_id):
    try:
        # Allow token from query string for img tags
        token = request.args.get('token') or request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({'error': 'No authorization provided'}), 401
        
        # Verify token
        try:
            user_id = auth_service.verify_token(token)
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
        
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get the thumbnail index (default to primary/middle one)
        thumbnail_index = request.args.get('index', type=int)
        
        if thumbnail_index is not None:
            # Get specific thumbnail by index
            thumbnails = video.outputs.get('thumbnails', [])
            if 0 <= thumbnail_index < len(thumbnails):
                thumbnail_path = thumbnails[thumbnail_index]
            else:
                return jsonify({'error': 'Thumbnail index out of range'}), 404
        else:
            # Get primary thumbnail
            thumbnail_path = video.outputs.get('thumbnail')
        
        if not thumbnail_path or not os.path.exists(thumbnail_path):
            return jsonify({'error': 'Thumbnail not found'}), 404
        
        return send_file(
            thumbnail_path,
            mimetype='image/jpeg',
            as_attachment=False
        )
        
    except Exception as e:
        logger.error(f"Get thumbnail error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/videos/<video_id>/thumbnails', methods=['GET'])
@require_auth
def get_all_thumbnails(user_id, video_id):
    try:
        video = video_service.get_video(video_id)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Check if user owns the video
        if str(video.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized'}), 403
        
        thumbnails = video.outputs.get('thumbnails', [])
        thumbnail_info = []
        
        for i, thumb_path in enumerate(thumbnails):
            if os.path.exists(thumb_path):
                thumbnail_info.append({
                    'index': i,
                    'url': f'/api/videos/{video_id}/thumbnail?index={i}',
                    'filename': os.path.basename(thumb_path)
                })
        
        return jsonify({
            'thumbnails': thumbnail_info,
            'primary_index': 2 if len(thumbnails) > 2 else 0,
            'count': len(thumbnail_info)
        }), 200
        
    except Exception as e:
        logger.error(f"Get thumbnails error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413

# Get current user info from JWT
from functools import wraps
import jwt

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(' ')[1]
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            user_id = data['user_id']
        except Exception as e:
            return jsonify({'error': 'Token is invalid!'}), 401
        return f(user_id, *args, **kwargs)
    return decorated

@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user(user_id):
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Convert ObjectId to string
    user['_id'] = str(user['_id'])
    
    # Remove password_hash from response (security)
    user.pop('password', None)
    user.pop('password_hash', None)
    
    # Convert any remaining bytes fields to strings if needed
    for key, value in user.items():
        if isinstance(value, bytes):
            user[key] = value.decode('utf-8') if value else None
    
    return jsonify(user), 200

@app.route('/api/auth/delete-account', methods=['DELETE'])
@require_auth
def delete_account(user_id):
    try:
        # Delete user account and all associated data
        auth_service.delete_user(user_id)
        
        logger.info(f"User account deleted successfully: {user_id}")
        return jsonify({
            'message': 'Account deleted successfully',
            'deleted_user_id': user_id
        }), 200
        
    except ValueError as e:
        logger.error(f"Delete account error for user {user_id}: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Unexpected error deleting account for user {user_id}")
        return jsonify({'error': 'Internal server error'}), 500

# ==================== ADMIN ROUTES ====================

def require_admin_auth(f):
    """Decorator for admin authentication"""
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'No admin token provided'}), 401
        
        token = auth_header.split(' ')[1]
        
        try:
            admin_id = auth_service.verify_token(token)
            if not admin_id:
                return jsonify({'error': 'Invalid admin token'}), 401
            
            # Check if user is admin
            admin = admin_service.get_admin_by_id(admin_id)
            if not admin:
                return jsonify({'error': 'Unauthorized - admin access required'}), 403
            
            return f(admin_id, *args, **kwargs)
        except Exception as e:
            logger.error(f"Admin auth error: {str(e)}")
            return jsonify({'error': 'Invalid admin token'}), 401
    
    decorated.__name__ = f.__name__
    return decorated

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Admin login endpoint"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        admin = admin_service.authenticate_admin(email, password)
        
        if not admin:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Generate JWT token
        token = auth_service.generate_token(str(admin._id))
        
        return jsonify({
            'token': token,
            'admin': admin.to_dict()
        }), 200
        
    except Exception as e:
        logger.exception("Admin login error")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/dashboard/stats', methods=['GET'])
@require_admin_auth
def get_dashboard_stats(admin_id):
    """Get dashboard statistics"""
    try:
        stats = admin_service.get_dashboard_stats()
        return jsonify(stats), 200
    except Exception as e:
        logger.exception("Error getting dashboard stats")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/analytics/user-growth', methods=['GET'])
@require_admin_auth
def get_user_growth(admin_id):
    """Get user growth data"""
    try:
        days = int(request.args.get('days', 30))
        data = admin_service.get_user_growth_data(days)
        return jsonify(data), 200
    except Exception as e:
        logger.exception("Error getting user growth")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/analytics/video-trends', methods=['GET'])
@require_admin_auth
def get_video_trends(admin_id):
    """Get video upload trends"""
    try:
        days = int(request.args.get('days', 30))
        data = admin_service.get_video_upload_trends(days)
        return jsonify(data), 200
    except Exception as e:
        logger.exception("Error getting video trends")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/analytics/video-status', methods=['GET'])
@require_admin_auth
def get_video_status_distribution(admin_id):
    """Get video status distribution"""
    try:
        data = admin_service.get_video_status_distribution()
        return jsonify(data), 200
    except Exception as e:
        logger.exception("Error getting video status distribution")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/analytics/content', methods=['GET'])
@require_admin_auth
def get_content_analytics(admin_id):
    """Get content analytics"""
    try:
        data = admin_service.get_content_analytics()
        return jsonify(data), 200
    except Exception as e:
        logger.exception("Error getting content analytics")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/users', methods=['GET'])
@require_admin_auth
def admin_get_users(admin_id):
    """Get all users with pagination"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        search = request.args.get('search', None)
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        result = admin_service.get_all_users(page, limit, search, sort_by, sort_order)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error getting users")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/users/<user_id>', methods=['GET'])
@require_admin_auth
def admin_get_user_details(admin_id, user_id):
    """Get detailed user information"""
    try:
        user = admin_service.get_user_details(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify(user), 200
    except Exception as e:
        logger.exception("Error getting user details")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/users/<user_id>', methods=['PUT'])
@require_admin_auth
def admin_update_user(admin_id, user_id):
    """Update user information"""
    try:
        updates = request.json
        
        success = admin_service.update_user(user_id, updates)
        
        if success:
            return jsonify({'message': 'User updated successfully'}), 200
        else:
            return jsonify({'error': 'Failed to update user'}), 400
    except Exception as e:
        logger.exception("Error updating user")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/users/<user_id>', methods=['DELETE'])
@require_admin_auth
def admin_delete_user(admin_id, user_id):
    """Delete user and all their data"""
    try:
        # Check permission
        admin = admin_service.get_admin_by_id(admin_id)
        if not admin.has_permission('delete_users'):
            return jsonify({'error': 'Permission denied'}), 403
        
        success = admin_service.delete_user(user_id)
        
        if success:
            return jsonify({'message': 'User deleted successfully'}), 200
        else:
            return jsonify({'error': 'Failed to delete user'}), 400
    except Exception as e:
        logger.exception("Error deleting user")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/users/<user_id>/toggle-status', methods=['POST'])
@require_admin_auth
def admin_toggle_user_status(admin_id, user_id):
    """Toggle user active/inactive status"""
    try:
        success = admin_service.toggle_user_status(user_id)
        
        if success:
            return jsonify({'message': 'User status updated successfully'}), 200
        else:
            return jsonify({'error': 'Failed to update user status'}), 400
    except Exception as e:
        logger.exception("Error toggling user status")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/videos', methods=['GET'])
@require_admin_auth
def admin_get_videos(admin_id):
    """Get all videos with pagination and filters"""
    try:
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        search = request.args.get('search', None)
        status_filter = request.args.get('status', None)
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        result = admin_service.get_all_videos(page, limit, search, status_filter, sort_by, sort_order)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Error getting videos")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/videos/<video_id>/logs', methods=['GET'])
@require_admin_auth
def admin_get_video_logs(admin_id, video_id):
    """Get video processing logs"""
    try:
        logs = admin_service.get_video_logs(video_id)
        
        if not logs:
            return jsonify({'error': 'Video not found'}), 404
        
        return jsonify(logs), 200
    except Exception as e:
        logger.exception("Error getting video logs")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/videos/<video_id>', methods=['DELETE'])
@require_admin_auth
def admin_delete_video(admin_id, video_id):
    """Delete video"""
    try:
        # Check permission
        admin = admin_service.get_admin_by_id(admin_id)
        if not admin.has_permission('delete_videos'):
            return jsonify({'error': 'Permission denied'}), 403
        
        success = admin_service.delete_video(video_id)
        
        if success:
            return jsonify({'message': 'Video deleted successfully'}), 200
        else:
            return jsonify({'error': 'Failed to delete video'}), 400
    except Exception as e:
        logger.exception("Error deleting video")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/admin/activity', methods=['GET'])
@require_admin_auth
def get_recent_activity(admin_id):
    """Get recent system activity"""
    try:
        limit = int(request.args.get('limit', 50))
        activities = admin_service.get_recent_activity(limit)
        return jsonify(activities), 200
    except Exception as e:
        logger.exception("Error getting recent activity")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)