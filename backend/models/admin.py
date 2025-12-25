"""
Admin Model
Handles admin user data structure and authentication
"""

from datetime import datetime
from bson.objectid import ObjectId

class Admin:
    """Admin user model for dashboard access and management"""
    
    def __init__(self, email, password_hash, name, role='admin', permissions=None, created_at=None, last_login=None, _id=None):
        self._id = _id if _id else ObjectId()
        self.email = email
        self.password_hash = password_hash
        self.name = name
        self.role = role  # 'super_admin', 'admin', 'moderator'
        self.permissions = permissions if permissions else self._get_default_permissions(role)
        self.created_at = created_at if created_at else datetime.now()
        self.last_login = last_login
        self.is_active = True
    
    def _get_default_permissions(self, role):
        """Get default permissions based on role"""
        if role == 'super_admin':
            return {
                'view_users': True,
                'edit_users': True,
                'delete_users': True,
                'view_videos': True,
                'delete_videos': True,
                'view_analytics': True,
                'manage_admins': True,
                'system_settings': True
            }
        elif role == 'admin':
            return {
                'view_users': True,
                'edit_users': True,
                'delete_users': False,
                'view_videos': True,
                'delete_videos': True,
                'view_analytics': True,
                'manage_admins': False,
                'system_settings': False
            }
        else:  # moderator
            return {
                'view_users': True,
                'edit_users': False,
                'delete_users': False,
                'view_videos': True,
                'delete_videos': False,
                'view_analytics': True,
                'manage_admins': False,
                'system_settings': False
            }
    
    def to_dict(self):
        """Convert admin object to dictionary"""
        return {
            '_id': str(self._id),
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'permissions': self.permissions,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'last_login': self.last_login.isoformat() if isinstance(self.last_login, datetime) else self.last_login,
            'is_active': self.is_active
        }
    
    @staticmethod
    def from_dict(data):
        """Create admin object from dictionary"""
        return Admin(
            email=data.get('email'),
            password_hash=data.get('password_hash'),
            name=data.get('name'),
            role=data.get('role', 'admin'),
            permissions=data.get('permissions'),
            created_at=data.get('created_at'),
            last_login=data.get('last_login'),
            _id=data.get('_id')
        )
    
    def has_permission(self, permission):
        """Check if admin has specific permission"""
        return self.permissions.get(permission, False)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now()
