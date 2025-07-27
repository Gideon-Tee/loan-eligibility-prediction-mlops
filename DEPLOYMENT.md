# Deployment Setup

## GitHub Secrets Required

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

### EC2_HOST
Your EC2 instance public IP or domain name
```
Example: 54.123.45.67
```

### EC2_USER
EC2 instance username
```
Example: ubuntu
```

### EC2_SSH_KEY
Your private SSH key content (the entire .pem file content)
```
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
-----END RSA PRIVATE KEY-----
```

## Initial EC2 Setup

1. **Clone repository on EC2:**
```bash
cd /home/ubuntu
git clone <your-repo-url> loan-eligibility
cd loan-eligibility
```

2. **Install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Setup environment:**
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. **Initialize Airflow:**
```bash
export AIRFLOW_HOME=/home/ubuntu/loan-eligibility/airflow
airflow db init
```

5. **Start services:**
```bash
./manage_services.sh start
```

## Usage

- **Deploy:** Push to main branch triggers automatic deployment
- **Manual deploy:** Use "Run workflow" in GitHub Actions
- **Service management:** Use `./manage_services.sh {start|stop|restart|status}`

## Health Checks

- Airflow: http://your-ec2-ip:8080
- MLflow: http://your-ec2-ip:5000  
- Model API: http://your-ec2-ip:5050/health