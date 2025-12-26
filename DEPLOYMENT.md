# 🚀 Deployment Guide - SnipX Video Editing SaaS

## Overview
- **Frontend**: Netlify
- **Backend**: Railway
- **Database**: MongoDB Atlas

---

## 📦 Prerequisites

1. **GitHub Account** (code repository)
2. **MongoDB Atlas Account** (free tier available)
3. **Railway Account** (sign up with GitHub)
4. **Netlify Account** (sign up with GitHub)

---

## 🗄️ STEP 1: MongoDB Atlas Setup

### 1.1 Create MongoDB Cluster
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Sign up/Login
3. Create New Project → Name it "SnipX" or "VideoSaaS"
4. Click **"Build a Database"**
5. Select **FREE** tier (M0 Sandbox)
6. Choose **AWS** provider, region closest to you
7. Cluster Name: `Cluster0` (default is fine)
8. Click **"Create Cluster"** (takes 3-5 minutes)

### 1.2 Configure Database Access
1. Go to **Database Access** (left sidebar)
2. Click **"Add New Database User"**
   - Username: `snipx_admin` (or your choice)
   - Password: Click **"Autogenerate Secure Password"** → **COPY THIS PASSWORD**
   - Database User Privileges: **"Read and write to any database"**
   - Click **"Add User"**

### 1.3 Configure Network Access
1. Go to **Network Access** (left sidebar)
2. Click **"Add IP Address"**
3. Click **"Allow Access from Anywhere"** (0.0.0.0/0)
   - Required for Railway backend to connect
4. Click **"Confirm"**

### 1.4 Get Connection String
1. Go to **Database** → Click **"Connect"** on your cluster
2. Select **"Connect your application"**
3. Driver: **Python**, Version: **3.12 or later**
4. Copy the connection string:
   ```
   mongodb+srv://snipx_admin:<password>@cluster0.xxxxx.mongodb.net/
   ```
5. **IMPORTANT**: Replace `<password>` with the password you copied earlier
6. Add database name at the end: `?retryWrites=true&w=majority`

**Final Connection String Example:**
```
mongodb+srv://snipx_admin:YourPassword123@cluster0.abc12.mongodb.net/?retryWrites=true&w=majority
```

**Save this connection string - you'll need it for Railway!**

---

## 🚂 STEP 2: Deploy Backend to Railway

### 2.1 Connect GitHub to Railway
1. Go to [Railway.app](https://railway.app)
2. Click **"Start a New Project"**
3. Select **"Deploy from GitHub repo"**
4. Authorize Railway to access your GitHub
5. Select your repository: **FYP** or **video-editing-saas**

### 2.2 Configure Railway Project
1. Railway will detect Python project automatically
2. Click on the deployed service
3. Go to **Settings** tab:
   - **Root Directory**: Set to `backend`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

### 2.3 Add Environment Variables
1. Go to **Variables** tab
2. Click **"New Variable"** and add these one by one:

```env
MONGODB_URI=mongodb+srv://snipx_admin:YourPassword123@cluster0.abc12.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=video_editing_saas
JWT_SECRET=your-super-secret-random-string-change-this-now
FLASK_ENV=production
PORT=5001
CORS_ORIGINS=https://your-site.netlify.app,http://localhost:5173
```

**IMPORTANT**:
- Replace `MONGODB_URI` with your Atlas connection string
- Change `JWT_SECRET` to a random secure string (use password generator)
- We'll update `CORS_ORIGINS` after deploying frontend

### 2.4 Get Backend URL
1. After deployment completes (2-3 minutes)
2. Go to **Settings** → **Domains**
3. Railway auto-generates a URL like: `https://your-backend-production.up.railway.app`
4. **COPY THIS URL** - you'll need it for frontend!

### 2.5 Test Backend
Open browser and test:
```
https://your-backend-production.up.railway.app/
```
Should see: `{"message": "SnipX API is running"}`

---

## 🌐 STEP 3: Deploy Frontend to Netlify

### 3.1 Update Frontend Environment
Before deploying, we need to set the backend URL:

1. Create `.env` file in project root (not in backend folder):
```env
VITE_API_URL=https://your-backend-production.up.railway.app
```
Replace with your actual Railway backend URL

### 3.2 Deploy to Netlify
1. Go to [Netlify](https://app.netlify.com)
2. Click **"Add new site"** → **"Import an existing project"**
3. Select **"Deploy with GitHub"**
4. Authorize Netlify → Select your repository
5. Configure build settings:
   - **Base directory**: Leave empty (root)
   - **Build command**: `npm run build`
   - **Publish directory**: `dist`
6. Click **"Show advanced"** → **"New variable"**
   - Key: `VITE_API_URL`
   - Value: `https://your-backend-production.up.railway.app`
7. Click **"Deploy site"**

### 3.3 Custom Domain (Optional)
1. After deployment, go to **Site settings** → **Domain management**
2. Click **"Add custom domain"** or use Netlify subdomain
3. Your site URL will be: `https://your-site-name.netlify.app`

### 3.4 Update CORS Settings
Now update Railway backend with frontend URL:
1. Go back to **Railway** → Your project → **Variables**
2. Update `CORS_ORIGINS`:
```env
CORS_ORIGINS=https://your-site-name.netlify.app
```
3. Railway will auto-redeploy

---

## ✅ STEP 4: Final Testing

### 4.1 Test Complete Flow
1. Open your Netlify site: `https://your-site-name.netlify.app`
2. Register a new user account
3. Login with credentials
4. Upload a video (small test video)
5. Try video processing features:
   - Subtitle generation
   - Audio enhancement (moderate level)
   - Thumbnail generation
6. Check if processed video downloads correctly

### 4.2 Verify Database
1. Go to MongoDB Atlas → **Database** → **Browse Collections**
2. You should see:
   - `users` collection (with your registered user)
   - `videos` collection (with uploaded video metadata)
   - `support_tickets` collection (empty initially)

### 4.3 Monitor Logs
- **Railway**: Go to **Deployments** → Click latest → View logs
- **Netlify**: Go to **Deploys** → Click latest → View function logs

---

## 🔧 Troubleshooting

### Backend Issues
**Problem**: `500 Internal Server Error`
- Check Railway logs for Python errors
- Verify MongoDB connection string is correct
- Ensure all environment variables are set

**Problem**: `CORS Error` in browser console
- Update `CORS_ORIGINS` in Railway to match Netlify URL
- Must include `https://` protocol

### Frontend Issues
**Problem**: "Network Error" or "Failed to fetch"
- Verify `VITE_API_URL` in Netlify environment variables
- Check if Railway backend is running (green status)
- Open backend URL directly to test

**Problem**: Build fails on Netlify
- Check build logs for missing dependencies
- Ensure `package.json` has all required packages
- Try `npm install` locally to verify

### Database Issues
**Problem**: "Connection timeout" to MongoDB
- Check Network Access in Atlas (should be 0.0.0.0/0)
- Verify connection string has correct password
- Ensure database user has read/write permissions

---

## 📝 Important Notes

1. **File Upload Limits**:
   - Railway: Default 100MB (upgrade plan for more)
   - Netlify Functions: 6MB (for API routes)
   - Use Railway backend for video uploads

2. **Processing Time**:
   - Audio enhancement: ~5-10 seconds per minute
   - Subtitle generation: ~15-30 seconds per minute
   - Railway free tier: 500 hours/month (sufficient for development)

3. **Cost Estimates** (Free Tiers):
   - MongoDB Atlas: 512MB storage (free forever)
   - Railway: $5 credit/month (no credit card required initially)
   - Netlify: 100GB bandwidth/month (free forever)

4. **Security**:
   - Change `JWT_SECRET` before production use
   - Never commit `.env` files to GitHub
   - Use strong passwords for MongoDB users
   - Enable 2FA on all services

---

## 🎉 Success Checklist

- [ ] MongoDB Atlas cluster created and connected
- [ ] Railway backend deployed and responding
- [ ] Netlify frontend deployed and accessible
- [ ] User registration/login working
- [ ] Video upload successful
- [ ] Audio enhancement processing works
- [ ] Subtitle generation works
- [ ] Thumbnail generation works
- [ ] Admin panel accessible

---

## 🆘 Need Help?

If you encounter issues:
1. Check deployment logs first (Railway/Netlify dashboards)
2. Verify all environment variables are correct
3. Test backend URL directly in browser
4. Check browser console for frontend errors

**Common Commands for Local Testing:**
```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py

# Frontend
npm install
npm run dev
```

---

**Deployment Date**: December 26, 2025
**Version**: 1.0.0
