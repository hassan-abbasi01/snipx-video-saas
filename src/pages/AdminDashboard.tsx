import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line, Doughnut } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);
import API_URL from '../config/api';

interface DashboardStats {
  total_users: number;
  active_users: number;
  total_videos: number;
  processing_videos: number;
  completed_videos: number;
  failed_videos: number;
  total_tickets: number;
  open_tickets: number;
  resolved_tickets: number;
  new_users_week: number;
  new_videos_week: number;
  total_storage_mb: number;
}

interface Activity {
  type: string;
  description: string;
  timestamp: string;
}

const AdminDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [userGrowth, setUserGrowth] = useState<any>(null);
  const [videoTrends, setVideoTrends] = useState<any>(null);
  const [videoStatus, setVideoStatus] = useState<any>(null);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('admin_token');
    return {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    };
  };

  const fetchDashboardData = async () => {
    try {
      const token = localStorage.getItem('admin_token');
      
      if (!token) {
        navigate('/admin/login');
        return;
      }

      // Fetch all dashboard data
      const [statsRes, growthRes, trendsRes, statusRes, activityRes] = await Promise.all([
        axios.get(`${API_URL}/api/admin/dashboard/stats`, getAuthHeaders()),
        axios.get(`${API_URL}/api/admin/analytics/user-growth?days=30`, getAuthHeaders()),
        axios.get(`${API_URL}/api/admin/analytics/video-trends?days=30`, getAuthHeaders()),
        axios.get(`${API_URL}/api/admin/analytics/video-status`, getAuthHeaders()),
        axios.get(`${API_URL}/api/admin/activity?limit=10`, getAuthHeaders())
      ]);

      setStats(statsRes.data);
      setUserGrowth(growthRes.data);
      setVideoTrends(trendsRes.data);
      setVideoStatus(statusRes.data);
      setActivities(activityRes.data);
      setLoading(false);
    } catch (error: any) {
      console.error('Error fetching dashboard data:', error);
      if (error.response?.status === 401 || error.response?.status === 403) {
        localStorage.removeItem('admin_token');
        navigate('/admin/login');
      }
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    navigate('/admin/login');
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading dashboard...</div>
      </div>
    );
  }

  // Chart configurations
  const userGrowthChart = {
    labels: userGrowth?.labels || [],
    datasets: [
      {
        label: 'New Users',
        data: userGrowth?.data || [],
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        tension: 0.4
      }
    ]
  };

  const videoTrendsChart = {
    labels: videoTrends?.labels || [],
    datasets: [
      {
        label: 'Videos Uploaded',
        data: videoTrends?.data || [],
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.5)',
        tension: 0.4
      }
    ]
  };

  const videoStatusChart = {
    labels: videoStatus?.labels || [],
    datasets: [
      {
        data: videoStatus?.data || [],
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',   // completed - green
          'rgba(251, 191, 36, 0.8)',  // processing - yellow
          'rgba(239, 68, 68, 0.8)',   // failed - red
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(251, 191, 36)',
          'rgb(239, 68, 68)',
        ],
        borderWidth: 1
      }
    ]
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-purple-50 relative overflow-hidden">
      {/* 3D Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-br from-purple-400/20 to-pink-400/20 rounded-full blur-xl animate-float-3d" />
        <div className="absolute bottom-32 right-20 w-40 h-40 bg-gradient-to-br from-blue-400/15 to-cyan-400/15 rounded-full blur-2xl animate-pulse-3d" />
      </div>

      {/* Header */}
      <header className="bg-white/90 backdrop-blur-md shadow-lg border-b border-white/20 relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Admin Dashboard</h1>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate('/admin/users')}
                className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 transition transform hover:scale-105"
              >
                Manage Users
              </button>
              <button
                onClick={() => navigate('/admin/videos')}
                className="px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl hover:from-green-600 hover:to-emerald-700 transition transform hover:scale-105"
              >
                Manage Videos
              </button>
              <button
                onClick={handleLogout}
                className="px-4 py-2 bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-xl hover:from-red-600 hover:to-pink-700 transition transform hover:scale-105"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Total Users */}
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20 transform hover:scale-105 hover:-translate-y-2 transition-all duration-300">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Total Users</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats?.total_users || 0}</p>
                <p className="text-green-600 text-sm mt-1">
                  +{stats?.new_users_week || 0} this week
                </p>
              </div>
              <div className="text-4xl">👥</div>
            </div>
          </div>

          {/* Total Videos */}
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20 transform hover:scale-105 hover:-translate-y-2 transition-all duration-300">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Total Videos</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats?.total_videos || 0}</p>
                <p className="text-green-600 text-sm mt-1">
                  +{stats?.new_videos_week || 0} this week
                </p>
              </div>
              <div className="text-4xl">🎬</div>
            </div>
          </div>

          {/* Active Users */}
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20 transform hover:scale-105 hover:-translate-y-2 transition-all duration-300">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Active Users (30d)</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stats?.active_users || 0}</p>
                <p className="text-gray-600 text-sm mt-1">
                  {stats?.total_users ? ((stats.active_users / stats.total_users) * 100).toFixed(1) : 0}% active
                </p>
              </div>
              <div className="text-4xl">✅</div>
            </div>
          </div>

          {/* Storage Used */}
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20 transform hover:scale-105 hover:-translate-y-2 transition-all duration-300">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Storage Used</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">
                  {stats?.total_storage_mb ? (stats.total_storage_mb / 1024).toFixed(2) : 0} GB
                </p>
                <p className="text-gray-600 text-sm mt-1">
                  {stats?.total_storage_mb?.toFixed(0) || 0} MB
                </p>
              </div>
              <div className="text-4xl">💾</div>
            </div>
          </div>
        </div>

        {/* Video Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-green-50 border-2 border-green-200 rounded-2xl p-6 transform hover:scale-105 transition-all duration-300">
            <p className="text-green-700 text-sm font-medium">Completed Videos</p>
            <p className="text-3xl font-bold text-green-800 mt-2">{stats?.completed_videos || 0}</p>
          </div>
          <div className="bg-yellow-50 border-2 border-yellow-200 rounded-2xl p-6 transform hover:scale-105 transition-all duration-300">
            <p className="text-yellow-700 text-sm font-medium">Processing Videos</p>
            <p className="text-3xl font-bold text-yellow-800 mt-2">{stats?.processing_videos || 0}</p>
          </div>
          <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6 transform hover:scale-105 transition-all duration-300">
            <p className="text-red-700 text-sm font-medium">Failed Videos</p>
            <p className="text-3xl font-bold text-red-800 mt-2">{stats?.failed_videos || 0}</p>
          </div>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* User Growth Chart */}
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">User Growth (30 Days)</h3>
            <Line data={userGrowthChart} options={{ responsive: true, maintainAspectRatio: true }} />
          </div>

          {/* Video Trends Chart */}
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Video Upload Trends (30 Days)</h3>
            <Line data={videoTrendsChart} options={{ responsive: true, maintainAspectRatio: true }} />
          </div>
        </div>

        {/* Video Status Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Video Status Distribution</h3>
            <Doughnut data={videoStatusChart} options={{ responsive: true, maintainAspectRatio: true }} />
          </div>

          {/* Recent Activity */}
          <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20 lg:col-span-2">
            <h3 className="text-xl font-semibold text-gray-800 mb-4">Recent Activity</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {activities.map((activity, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl border border-purple-100 hover:shadow-md transition">
                  <div className="text-2xl">
                    {activity.type === 'user_registration' ? '👤' : '🎬'}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-700">{activity.description}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(activity.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              ))}
              {activities.length === 0 && (
                <p className="text-gray-500 text-center py-4">No recent activity</p>
              )}
            </div>
          </div>
        </div>

        {/* Support Tickets */}
        <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 shadow-xl border border-white/20">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Support Tickets</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gradient-to-br from-slate-50 to-gray-100 rounded-xl p-4 border border-gray-200">
              <p className="text-gray-600 text-sm font-medium">Total Tickets</p>
              <p className="text-2xl font-bold text-gray-800 mt-2">{stats?.total_tickets || 0}</p>
            </div>
            <div className="bg-yellow-50 border-2 border-yellow-200 rounded-xl p-4">
              <p className="text-yellow-700 text-sm font-medium">Open Tickets</p>
              <p className="text-2xl font-bold text-yellow-800 mt-2">{stats?.open_tickets || 0}</p>
            </div>
            <div className="bg-green-50 border-2 border-green-200 rounded-xl p-4">
              <p className="text-green-700 text-sm font-medium">Resolved Tickets</p>
              <p className="text-2xl font-bold text-green-800 mt-2">{stats?.resolved_tickets || 0}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
