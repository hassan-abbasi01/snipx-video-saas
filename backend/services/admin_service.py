"""
Admin Service
Handles admin operations: user management, video logs, analytics, content moderation
"""

from datetime import datetime, timedelta
from bson.objectid import ObjectId
from models.admin import Admin
import bcrypt

class AdminService:
    """Service for admin dashboard operations and analytics"""
    
    def __init__(self, db):
        self.db = db
        self.admins = db.admins
        self.users = db.users
        self.videos = db.videos
        self.support_tickets = db.support_tickets
    
    # ==================== ADMIN AUTHENTICATION ====================
    
    def create_admin(self, email, password, name, role='admin'):
        """Create a new admin user"""
        # Check if admin already exists
        existing = self.admins.find_one({'email': email})
        if existing:
            raise ValueError('Admin with this email already exists')
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Create admin
        admin = Admin(
            email=email,
            password_hash=password_hash,
            name=name,
            role=role
        )
        
        # Save to database
        result = self.admins.insert_one({
            'email': admin.email,
            'password_hash': admin.password_hash,
            'name': admin.name,
            'role': admin.role,
            'permissions': admin.permissions,
            'created_at': admin.created_at,
            'last_login': None,
            'is_active': True
        })
        
        admin._id = result.inserted_id
        return admin
    
    def authenticate_admin(self, email, password):
        """Authenticate admin and return admin object"""
        admin_data = self.admins.find_one({'email': email})
        
        if not admin_data:
            return None
        
        if not admin_data.get('is_active', True):
            return None
        
        # Check password
        if not bcrypt.checkpw(password.encode('utf-8'), admin_data['password_hash']):
            return None
        
        # Update last login
        self.admins.update_one(
            {'_id': admin_data['_id']},
            {'$set': {'last_login': datetime.now()}}
        )
        
        return Admin.from_dict(admin_data)
    
    def get_admin_by_id(self, admin_id):
        """Get admin by ID"""
        admin_data = self.admins.find_one({'_id': ObjectId(admin_id)})
        if admin_data:
            return Admin.from_dict(admin_data)
        return None
    
    # ==================== DASHBOARD ANALYTICS ====================
    
    def get_dashboard_stats(self):
        """Get overall dashboard statistics"""
        try:
            # Total counts
            total_users = self.users.count_documents({})
            total_videos = self.videos.count_documents({})
            total_tickets = self.support_tickets.count_documents({})
            
            # Active users (logged in last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            active_users = self.users.count_documents({
                'last_login': {'$gte': thirty_days_ago}
            })
            
            # Video statistics
            processing_videos = self.videos.count_documents({'status': 'processing'})
            completed_videos = self.videos.count_documents({'status': 'completed'})
            failed_videos = self.videos.count_documents({'status': 'failed'})
            
            # Support ticket statistics
            open_tickets = self.support_tickets.count_documents({'status': 'open'})
            resolved_tickets = self.support_tickets.count_documents({'status': 'resolved'})
            
            # Recent activity (last 7 days)
            seven_days_ago = datetime.now() - timedelta(days=7)
            new_users_week = self.users.count_documents({
                'created_at': {'$gte': seven_days_ago}
            })
            new_videos_week = self.videos.count_documents({
                'created_at': {'$gte': seven_days_ago}
            })
            
            # Storage usage (in MB)
            total_storage = 0
            for video in self.videos.find({}, {'size': 1}):
                total_storage += video.get('size', 0)
            total_storage_mb = total_storage / (1024 * 1024)
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'total_videos': total_videos,
                'processing_videos': processing_videos,
                'completed_videos': completed_videos,
                'failed_videos': failed_videos,
                'total_tickets': total_tickets,
                'open_tickets': open_tickets,
                'resolved_tickets': resolved_tickets,
                'new_users_week': new_users_week,
                'new_videos_week': new_videos_week,
                'total_storage_mb': round(total_storage_mb, 2),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting dashboard stats: {e}")
            return {}
    
    def get_user_growth_data(self, days=30):
        """Get user registration growth over time"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            pipeline = [
                {
                    '$match': {
                        'created_at': {'$gte': start_date, '$lte': end_date}
                    }
                },
                {
                    '$group': {
                        '_id': {
                            '$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}
                        },
                        'count': {'$sum': 1}
                    }
                },
                {
                    '$sort': {'_id': 1}
                }
            ]
            
            results = list(self.users.aggregate(pipeline))
            
            return {
                'labels': [r['_id'] for r in results],
                'data': [r['count'] for r in results]
            }
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting user growth: {e}")
            return {'labels': [], 'data': []}
    
    def get_video_upload_trends(self, days=30):
        """Get video upload trends over time"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            pipeline = [
                {
                    '$match': {
                        'created_at': {'$gte': start_date, '$lte': end_date}
                    }
                },
                {
                    '$group': {
                        '_id': {
                            '$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}
                        },
                        'count': {'$sum': 1}
                    }
                },
                {
                    '$sort': {'_id': 1}
                }
            ]
            
            results = list(self.videos.aggregate(pipeline))
            
            return {
                'labels': [r['_id'] for r in results],
                'data': [r['count'] for r in results]
            }
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting video trends: {e}")
            return {'labels': [], 'data': []}
    
    def get_video_status_distribution(self):
        """Get distribution of video statuses"""
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$status',
                        'count': {'$sum': 1}
                    }
                }
            ]
            
            results = list(self.videos.aggregate(pipeline))
            
            return {
                'labels': [r['_id'] for r in results],
                'data': [r['count'] for r in results]
            }
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting video distribution: {e}")
            return {'labels': [], 'data': []}
    
    # ==================== USER MANAGEMENT ====================
    
    def get_all_users(self, page=1, limit=20, search=None, sort_by='created_at', sort_order='desc'):
        """Get all users with pagination and search"""
        try:
            skip = (page - 1) * limit
            
            # Build query
            query = {}
            if search:
                query['$or'] = [
                    {'email': {'$regex': search, '$options': 'i'}},
                    {'name': {'$regex': search, '$options': 'i'}}
                ]
            
            # Sort order
            sort_direction = -1 if sort_order == 'desc' else 1
            
            # Get users
            users = list(self.users.find(query, {'password_hash': 0})
                        .sort(sort_by, sort_direction)
                        .skip(skip)
                        .limit(limit))
            
            # Convert ObjectId to string
            for user in users:
                user['_id'] = str(user['_id'])
                if 'created_at' in user and isinstance(user['created_at'], datetime):
                    user['created_at'] = user['created_at'].isoformat()
                if 'last_login' in user and isinstance(user['last_login'], datetime):
                    user['last_login'] = user['last_login'].isoformat()
            
            # Get total count
            total = self.users.count_documents(query)
            
            return {
                'users': users,
                'total': total,
                'page': page,
                'pages': (total + limit - 1) // limit
            }
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting users: {e}")
            return {'users': [], 'total': 0, 'page': 1, 'pages': 0}
    
    def get_user_details(self, user_id):
        """Get detailed user information including videos"""
        try:
            user = self.users.find_one({'_id': ObjectId(user_id)}, {'password_hash': 0})
            
            if not user:
                return None
            
            # Convert ObjectId
            user['_id'] = str(user['_id'])
            if 'created_at' in user and isinstance(user['created_at'], datetime):
                user['created_at'] = user['created_at'].isoformat()
            if 'last_login' in user and isinstance(user['last_login'], datetime):
                user['last_login'] = user['last_login'].isoformat()
            
            # Get user's videos
            videos = list(self.videos.find({'user_id': user_id}))
            for video in videos:
                video['_id'] = str(video['_id'])
                if 'created_at' in video and isinstance(video['created_at'], datetime):
                    video['created_at'] = video['created_at'].isoformat()
            
            user['videos'] = videos
            user['video_count'] = len(videos)
            
            # Get user's support tickets
            tickets = list(self.support_tickets.find({'user_id': user_id}))
            for ticket in tickets:
                ticket['_id'] = str(ticket['_id'])
                if 'created_at' in ticket and isinstance(ticket['created_at'], datetime):
                    ticket['created_at'] = ticket['created_at'].isoformat()
            
            user['tickets'] = tickets
            user['ticket_count'] = len(tickets)
            
            return user
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting user details: {e}")
            return None
    
    def update_user(self, user_id, updates):
        """Update user information"""
        try:
            # Don't allow password_hash update through this method
            if 'password_hash' in updates:
                del updates['password_hash']
            
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': updates}
            )
            
            return result.modified_count > 0
        except Exception as e:
            print(f"[ADMIN SERVICE] Error updating user: {e}")
            return False
    
    def delete_user(self, user_id):
        """Delete user and all their data"""
        try:
            # Delete user's videos first
            self.videos.delete_many({'user_id': user_id})
            
            # Delete user's tickets
            self.support_tickets.delete_many({'user_id': user_id})
            
            # Delete user
            result = self.users.delete_one({'_id': ObjectId(user_id)})
            
            return result.deleted_count > 0
        except Exception as e:
            print(f"[ADMIN SERVICE] Error deleting user: {e}")
            return False
    
    def toggle_user_status(self, user_id):
        """Toggle user active/inactive status"""
        try:
            user = self.users.find_one({'_id': ObjectId(user_id)})
            if not user:
                return False
            
            new_status = not user.get('is_active', True)
            
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$set': {'is_active': new_status}}
            )
            
            return result.modified_count > 0
        except Exception as e:
            print(f"[ADMIN SERVICE] Error toggling user status: {e}")
            return False
    
    # ==================== VIDEO MANAGEMENT ====================
    
    def get_all_videos(self, page=1, limit=20, search=None, status_filter=None, sort_by='created_at', sort_order='desc'):
        """Get all videos with pagination, search, and filters"""
        try:
            skip = (page - 1) * limit
            
            # Build query
            query = {}
            if search:
                query['filename'] = {'$regex': search, '$options': 'i'}
            
            if status_filter:
                query['status'] = status_filter
            
            # Sort order
            sort_direction = -1 if sort_order == 'desc' else 1
            
            # Get videos
            videos = list(self.videos.find(query)
                         .sort(sort_by, sort_direction)
                         .skip(skip)
                         .limit(limit))
            
            # Convert ObjectId and add user info
            for video in videos:
                video['_id'] = str(video['_id'])
                if 'created_at' in video and isinstance(video['created_at'], datetime):
                    video['created_at'] = video['created_at'].isoformat()
                if 'processed_at' in video and isinstance(video['processed_at'], datetime):
                    video['processed_at'] = video['processed_at'].isoformat()
                
                # Get user info
                user = self.users.find_one({'_id': ObjectId(video.get('user_id'))}, {'email': 1, 'name': 1})
                if user:
                    video['user_email'] = user.get('email', 'Unknown')
                    video['user_name'] = user.get('name', 'Unknown')
            
            # Get total count
            total = self.videos.count_documents(query)
            
            return {
                'videos': videos,
                'total': total,
                'page': page,
                'pages': (total + limit - 1) // limit
            }
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting videos: {e}")
            return {'videos': [], 'total': 0, 'page': 1, 'pages': 0}
    
    def get_video_logs(self, video_id):
        """Get processing logs for a specific video"""
        try:
            video = self.videos.find_one({'_id': ObjectId(video_id)})
            
            if not video:
                return None
            
            video['_id'] = str(video['_id'])
            if 'created_at' in video and isinstance(video['created_at'], datetime):
                video['created_at'] = video['created_at'].isoformat()
            if 'processed_at' in video and isinstance(video['processed_at'], datetime):
                video['processed_at'] = video['processed_at'].isoformat()
            
            # Get user info
            user = self.users.find_one({'_id': ObjectId(video.get('user_id'))})
            if user:
                video['user'] = {
                    'email': user.get('email'),
                    'name': user.get('name')
                }
            
            return video
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting video logs: {e}")
            return None
    
    def delete_video(self, video_id):
        """Delete video and its files"""
        try:
            video = self.videos.find_one({'_id': ObjectId(video_id)})
            
            if not video:
                return False
            
            # Delete video file
            import os
            if 'filepath' in video and os.path.exists(video['filepath']):
                os.remove(video['filepath'])
            
            # Delete from database
            result = self.videos.delete_one({'_id': ObjectId(video_id)})
            
            return result.deleted_count > 0
        except Exception as e:
            print(f"[ADMIN SERVICE] Error deleting video: {e}")
            return False
    
    # ==================== CONTENT ANALYTICS ====================
    
    def get_content_analytics(self):
        """Get content analytics and insights"""
        try:
            # Top users by video count
            top_users_pipeline = [
                {
                    '$group': {
                        '_id': '$user_id',
                        'video_count': {'$sum': 1}
                    }
                },
                {'$sort': {'video_count': -1}},
                {'$limit': 10}
            ]
            
            top_users_raw = list(self.videos.aggregate(top_users_pipeline))
            
            top_users = []
            for item in top_users_raw:
                user = self.users.find_one({'_id': ObjectId(item['_id'])}, {'email': 1, 'name': 1})
                if user:
                    top_users.append({
                        'user_id': str(item['_id']),
                        'email': user.get('email', 'Unknown'),
                        'name': user.get('name', 'Unknown'),
                        'video_count': item['video_count']
                    })
            
            # Average processing time
            completed_videos = list(self.videos.find(
                {'status': 'completed', 'created_at': {'$exists': True}, 'processed_at': {'$exists': True}},
                {'created_at': 1, 'processed_at': 1}
            ))
            
            processing_times = []
            for video in completed_videos:
                if isinstance(video.get('created_at'), datetime) and isinstance(video.get('processed_at'), datetime):
                    diff = (video['processed_at'] - video['created_at']).total_seconds()
                    processing_times.append(diff)
            
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Most common video formats
            format_pipeline = [
                {
                    '$group': {
                        '_id': '$format',
                        'count': {'$sum': 1}
                    }
                },
                {'$sort': {'count': -1}},
                {'$limit': 5}
            ]
            
            formats = list(self.videos.aggregate(format_pipeline))
            
            return {
                'top_users': top_users,
                'avg_processing_time_seconds': round(avg_processing_time, 2),
                'video_formats': [{'format': f['_id'], 'count': f['count']} for f in formats]
            }
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting content analytics: {e}")
            return {}
    
    # ==================== SYSTEM LOGS ====================
    
    def get_recent_activity(self, limit=50):
        """Get recent system activity"""
        try:
            activities = []
            
            # Recent user registrations
            recent_users = list(self.users.find({}, {'email': 1, 'created_at': 1})
                               .sort('created_at', -1)
                               .limit(10))
            
            for user in recent_users:
                activities.append({
                    'type': 'user_registration',
                    'description': f"New user registered: {user.get('email')}",
                    'timestamp': user.get('created_at')
                })
            
            # Recent video uploads
            recent_videos = list(self.videos.find({}, {'filename': 1, 'user_id': 1, 'created_at': 1})
                                .sort('created_at', -1)
                                .limit(10))
            
            for video in recent_videos:
                user = self.users.find_one({'_id': ObjectId(video.get('user_id'))}, {'email': 1})
                user_email = user.get('email', 'Unknown') if user else 'Unknown'
                activities.append({
                    'type': 'video_upload',
                    'description': f"{user_email} uploaded {video.get('filename')}",
                    'timestamp': video.get('created_at')
                })
            
            # Sort by timestamp
            activities.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min, reverse=True)
            
            # Convert timestamps
            for activity in activities[:limit]:
                if isinstance(activity['timestamp'], datetime):
                    activity['timestamp'] = activity['timestamp'].isoformat()
            
            return activities[:limit]
        except Exception as e:
            print(f"[ADMIN SERVICE] Error getting recent activity: {e}")
            return []
