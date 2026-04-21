#!/usr/bin/env bash
# Polls for RTX 4000 Ada availability across DO regions.
# When a slot opens, creates the GPU droplet, migrates the server, and destroys the CPU droplet.
# Runs every 30 min via cron on the CPU droplet.

set -e

DO_TOKEN="dop_v1_fde770d7f915389b6b1be55e2eee30483452aecda2e1cfddab321915d36dd1bc"
SSH_KEY_ID="55527916"
GPU_SIZE="gpu-4000adax1-20gb"
REGIONS="nyc3 sfo3 sgp1 lon1 ams3 fra1 tor1 syd1 ric1 atl1"
LOG="/root/gpu_poller.log"
DONE_FLAG="/root/.gpu_provisioned"

log() { echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') $*" | tee -a "$LOG"; }

# Don't run if GPU already provisioned
if [ -f "$DONE_FLAG" ]; then
  exit 0
fi

log "Polling for $GPU_SIZE availability..."

api() { curl -sf -H "Authorization: Bearer $DO_TOKEN" -H "Content-Type: application/json" "$@"; }

for region in $REGIONS; do
  log "Trying $region..."

  RESPONSE=$(api -X POST "https://api.digitalocean.com/v2/droplets" \
    -d "{
      \"name\": \"mindmate-gpu\",
      \"region\": \"$region\",
      \"size\": \"$GPU_SIZE\",
      \"image\": \"ubuntu-22-04-x64\",
      \"ssh_keys\": [$SSH_KEY_ID]
    }" 2>/dev/null || true)

  GPU_ID=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['droplet']['id'])" 2>/dev/null || true)

  if [ -z "$GPU_ID" ]; then
    log "$region: unavailable"
    continue
  fi

  log "$region: GPU droplet created (id=$GPU_ID) — waiting for active status..."

  # Wait up to 5 min for droplet to become active
  for i in $(seq 1 30); do
    sleep 10
    STATUS=$(api "https://api.digitalocean.com/v2/droplets/$GPU_ID" \
      | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['droplet']['status'])" 2>/dev/null || echo "unknown")
    GPU_IP=$(api "https://api.digitalocean.com/v2/droplets/$GPU_ID" \
      | python3 -c "
import json,sys
d=json.load(sys.stdin)['droplet']
nets=[n['ip_address'] for n in d['networks']['v4'] if n['type']=='public']
print(nets[0] if nets else '')
" 2>/dev/null || true)
    log "  status=$STATUS ip=$GPU_IP"
    if [ "$STATUS" = "active" ] && [ -n "$GPU_IP" ]; then
      break
    fi
  done

  if [ "$STATUS" != "active" ]; then
    log "ERROR: GPU droplet never became active — aborting"
    exit 1
  fi

  log "GPU is active at $GPU_IP — waiting for SSH..."
  for i in $(seq 1 18); do
    sleep 10
    if ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$GPU_IP "echo ok" &>/dev/null; then
      log "SSH ready"
      break
    fi
  done

  log "Installing CUDA deps on GPU droplet..."
  ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@$GPU_IP bash << 'REMOTE'
set -e
# Wait for cloud-init
cloud-init status --wait 2>/dev/null || true
DEBIAN_FRONTEND=noninteractive apt update -qq
DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-pip python3-venv nginx cmake build-essential git rsync

# CUDA llama-cpp-python
python3 -m venv ~/venvs/mindmate
source ~/venvs/mindmate/bin/activate
pip install --upgrade pip -q
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
pip install fastapi "uvicorn[standard]" python-multipart -q
REMOTE

  log "Syncing code and model to GPU droplet..."
  rsync -az --progress -e "ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no" \
    /root/mindmate/ root@$GPU_IP:~/mindmate/

  log "Setting N_GPU_LAYERS=-1 in config..."
  ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@$GPU_IP \
    "sed -i 's/N_GPU_LAYERS = 0/N_GPU_LAYERS = -1/' ~/mindmate/deploy/config.py"

  log "Setting up systemd and nginx on GPU droplet..."
  ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@$GPU_IP bash << REMOTE
cat > /etc/systemd/system/mindmate.service << 'SVC'
[Unit]
Description=MindMate / Anchor FastAPI server
After=network.target

[Service]
User=root
WorkingDirectory=/root/mindmate/deploy
ExecStart=/root/mindmate/deploy/start.sh
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
SVC

cat > /etc/nginx/sites-available/mindmate << 'NGX'
server {
    listen 80;
    server_name $GPU_IP;
    client_max_body_size 1M;
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
        add_header X-Accel-Buffering no;
    }
}
NGX

ln -sf /etc/nginx/sites-available/mindmate /etc/nginx/sites-enabled/mindmate
rm -f /etc/nginx/sites-enabled/default
chmod +x /root/mindmate/deploy/start.sh
systemctl daemon-reload
systemctl enable mindmate
systemctl start mindmate
nginx -t && systemctl restart nginx
REMOTE

  log "GPU droplet fully configured at http://$GPU_IP"

  # Copy billing guard to GPU droplet
  scp -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
    /root/billing_guard.sh root@$GPU_IP:/root/billing_guard.sh
  ssh -i /root/.ssh/id_ed25519 -o StrictHostKeyChecking=no root@$GPU_IP \
    "chmod +x /root/billing_guard.sh && (crontab -l 2>/dev/null; echo '0 * * * * /root/billing_guard.sh') | crontab -"

  # Get CPU droplet ID and destroy it
  CPU_ID=$(api "https://api.digitalocean.com/v2/droplets?per_page=100" \
    | python3 -c "
import json,sys
droplets=json.load(sys.stdin)['droplets']
for d in droplets:
    if d['name']=='mindmate-vm':
        print(d['id'])
" 2>/dev/null || true)

  if [ -n "$CPU_ID" ]; then
    log "Destroying CPU droplet (id=$CPU_ID)..."
    api -X DELETE "https://api.digitalocean.com/v2/droplets/$CPU_ID"
    log "CPU droplet destroyed."
  fi

  echo "$GPU_IP" > "$DONE_FLAG"
  log "Done. New server: http://$GPU_IP"

  # Send notification to log (user can check /root/gpu_poller.log)
  wall "MindMate GPU droplet is live at http://$GPU_IP" 2>/dev/null || true
  break
done

if [ ! -f "$DONE_FLAG" ]; then
  log "No GPU availability found this run. Will retry in 30 min."
fi
