# Deployment Guide - Production-Ready Streamlit App

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended - FREE)

**Pros:**
- ✅ Completely free
- ✅ Automatic HTTPS
- ✅ Easy updates (push to GitHub)
- ✅ Built-in authentication
- ✅ No DevOps knowledge needed

**Steps:**
1. **Prepare Repository**
   ```bash
   git add .
   git commit -m "feat: production-ready Streamlit app"
   git push origin main
   ```

2. **Deploy**
   - Visit: https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `Regime Detection & Strategy`
   - Main file: `app.py`
   - Click "Deploy"

3. **Configure** (Optional)
   - Add secrets in Streamlit Cloud dashboard
   - Set custom subdomain
   - Configure authentication

4. **Access**
   - Your app: `https://your-username-regime-detection.streamlit.app`
   - Share URL with anyone

**Limitations:**
- 1 GB RAM per app
- Sleeps after inactivity (wakes in ~30s)
- Public by default (private requires paid plan)

---

### 2. Heroku (FREE Tier Available)

**Pros:**
- ✅ Always-on (no sleep)
- ✅ More resources
- ✅ Custom domains
- ✅ Add-ons (Redis, Postgres)

**Steps:**
1. **Install Heroku CLI**
   ```bash
   brew install heroku/brew/heroku
   ```

2. **Login**
   ```bash
   heroku login
   ```

3. **Create App**
   ```bash
   heroku create regime-detection-app
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Open**
   ```bash
   heroku open
   ```

**Configuration:**
- `Procfile` ✅ Already created
- `runtime.txt` ✅ Already created
- `requirements.txt` ✅ Already configured

**Cost:**
- Free tier: 550 dyno hours/month
- Hobby: $7/month (always-on)
- Production: $25+/month (more resources)

---

### 3. Docker + Cloud Provider (AWS/GCP/Azure)

**Pros:**
- ✅ Full control
- ✅ Scalable
- ✅ Professional infrastructure
- ✅ VPC, load balancers, auto-scaling

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Deploy to AWS ECS:**
```bash
# Build
docker build -t regime-detection .

# Tag
docker tag regime-detection:latest YOUR_ECR_URL/regime-detection:latest

# Push
docker push YOUR_ECR_URL/regime-detection:latest

# Deploy to ECS
aws ecs update-service --cluster your-cluster --service regime-detection --force-new-deployment
```

**Deploy to GCP Cloud Run:**
```bash
gcloud run deploy regime-detection \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

**Cost:**
- AWS: ~$20-50/month (t3.small)
- GCP: ~$15-40/month (Cloud Run)
- Azure: ~$25-60/month (App Service)

---

### 4. Railway (Modern Alternative)

**Pros:**
- ✅ Easy like Streamlit Cloud
- ✅ More generous free tier
- ✅ Better performance
- ✅ Persistent storage

**Steps:**
1. Visit: https://railway.app
2. "New Project" → "Deploy from GitHub"
3. Select repository
4. Railway auto-detects and deploys
5. Access via provided URL

**Cost:**
- Free: $5 credit/month
- Developer: $5/month + usage
- Team: $20/month + usage

---

## 🔒 Security Best Practices

### 1. Environment Variables
Never commit secrets. Use `.env` file locally:

```bash
# .env (DO NOT COMMIT)
ALPHA_VANTAGE_API_KEY=your_key_here
SECRET_KEY=your_secret
```

**Streamlit Cloud:**
- Settings → Secrets → Add TOML format

**Heroku:**
```bash
heroku config:set ALPHA_VANTAGE_API_KEY=your_key
```

### 2. Authentication (Optional)
Add password protection:

```python
# Add to app.py
import streamlit as st

def check_password():
    """Returns `True` if user had correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

if check_password():
    # Your app code here
    st.title("App")
```

### 3. Rate Limiting
Prevent API abuse:

```python
# Add to pages
import time
from streamlit import session_state as ss

if 'last_request' not in ss:
    ss.last_request = 0

if time.time() - ss.last_request < 5:  # 5 second cooldown
    st.warning("Please wait before refreshing")
    st.stop()

ss.last_request = time.time()
```

---

## ⚡ Performance Optimization

### 1. Caching
Already implemented in data loader:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(ticker, start, end):
    return loader.load_data(ticker, start, end)
```

### 2. Resource Limits
```toml
# .streamlit/config.toml
[server]
maxUploadSize = 200
maxMessageSize = 200

[runner]
magicEnabled = false
```

### 3. Optimize Large Datasets
```python
# Use chunking for large data
if len(data) > 10000:
    data = data.tail(10000)  # Limit to recent data
```

---

## 📊 Monitoring

### Streamlit Cloud
- Built-in analytics in dashboard
- View app logs
- Monitor resource usage

### Heroku
```bash
# View logs
heroku logs --tail

# Monitor resources
heroku ps

# Add-ons
heroku addons:create papertrail  # Logging
```

### Custom Monitoring
```python
# Add to app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"User loaded {ticker}")
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/
```

---

## 🎯 Post-Deployment Checklist

- [ ] App loads without errors
- [ ] Data fetching works (test with SPY)
- [ ] All 4 pages accessible
- [ ] Charts render correctly
- [ ] Regime detection completes
- [ ] Strategy analysis runs
- [ ] Mobile responsive
- [ ] HTTPS enabled
- [ ] Domain configured (if custom)
- [ ] Analytics enabled
- [ ] Error logging works
- [ ] Backup strategy in place

---

## 🆘 Troubleshooting

### Issue: Out of Memory
**Solution:**
- Reduce date range
- Use downsampled data
- Increase dyno size (Heroku)
- Upgrade tier (Streamlit Cloud)

### Issue: Slow Loading
**Solution:**
- Enable caching
- Optimize queries
- Use CDN for static assets
- Reduce feature count

### Issue: App Crashes
**Solution:**
- Check logs
- Add error handling
- Test locally first
- Use smaller datasets for testing

---

## 📈 Scaling Strategy

**Phase 1: Prototype (0-100 users)**
- Streamlit Cloud free tier
- Basic caching
- Single region

**Phase 2: Growth (100-1,000 users)**
- Heroku Hobby dyno
- Redis caching
- CDN for assets

**Phase 3: Scale (1,000+ users)**
- AWS/GCP with auto-scaling
- Database for caching
- Multi-region deployment
- Load balancer

---

## 💰 Cost Estimates

| Platform | Free Tier | Basic | Production |
|----------|-----------|-------|------------|
| **Streamlit Cloud** | ✅ Free | - | $25/mo |
| **Heroku** | 550 hrs/mo | $7/mo | $25+/mo |
| **Railway** | $5 credit | $5/mo | $20+/mo |
| **AWS** | 12 months | $20/mo | $50+/mo |
| **GCP** | $300 credit | $15/mo | $40+/mo |

---

## 🎉 Recommended Setup

**For personal use:**
→ Streamlit Cloud (Free)

**For small team:**
→ Heroku Hobby ($7/mo)

**For production:**
→ AWS ECS or GCP Cloud Run ($30-50/mo)

---

**Need help?** Check logs first, then GitHub issues.