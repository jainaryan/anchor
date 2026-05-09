# Anchor — Deployment Guide

---

## Current Production (DigitalOcean) ✅

**URL:** https://tryanchor.me  
**Server:** DigitalOcean c-4 dedicated CPU droplet, IP: 209.38.122.228  
**Model:** `exports/mindmate_llama_sft_ck1600/` (ck1600 GGUF — ⚠️ should upgrade to `mindmate_genzv2_ck1200_q4_k_m.gguf`)  
**Stack:** FastAPI + llama-cpp-python (CPU, N_GPU_LAYERS=0), nginx reverse proxy, systemd service  

```bash
# SSH
ssh -i ~/.ssh/id_ed25519 root@209.38.122.228

# Restart service
systemctl restart mindmate

# Tail logs
journalctl -u mindmate -f

# Deploy static file changes
rsync -az -e "ssh -i ~/.ssh/id_ed25519" deploy/static/ root@209.38.122.228:~/mindmate/deploy/static/

# ⚠️ Upgrade model to genzv2_ck1200 (better benchmark scores)
rsync -az exports/mindmate_genzv2_ck1200_q4_k_m.gguf root@209.38.122.228:~/mindmate/exports/
# Then update deploy/config.py MODEL_PATH and restart
```

**Billing guard:** `/root/billing_guard.sh` — hourly cron, destroys droplet at $175/month spend.

---

## New Deployment Guide (Azure)

Target: Azure Standard_D4as_v5 (4 vCPU, 16 GB RAM), Central India, 32 GB Standard SSD.  
Cost: ~$0.19/hr on-demand. Use **Azure for Students** ($100/yr free with NUS `.edu.sg` email — no credit card).  
*(The existing production server is DigitalOcean — use this section if provisioning a new/replacement server.)*

---

## Phase 1 — Azure account setup (once, manual)

1. Sign up at [azure.microsoft.com/en-us/free/students](https://azure.microsoft.com/en-us/free/students) with your NUS `.edu.sg` email.
2. Run `az login` in your terminal to authenticate the Azure CLI.
3. Confirm: `az account show`

---

## Phase 2 — Provision the VM (via Azure MCP in Claude Code)

Start a Claude Code session with Azure MCP connected and ask it to run each step.

**a. Create resource group**
> "Create a resource group called `mindmate-rg` in `centralindia`"

**b. Create VM**
> "Create a VM called `mindmate-vm` in `mindmate-rg`, size `Standard_D4as_v5`, image `Ubuntu2204`, region `centralindia`, generate SSH keys, open ports 22, 80, 443"

Azure MCP will return the **public IP** and SSH key path (typically `~/.ssh/mindmate-vm_key.pem`).

**c. Verify disk**
> "Show me the disk attached to mindmate-vm"  
Should be 32 GB Standard SSD by default.

---

## Phase 3 — Server setup (via SSH)

```bash
ssh -i ~/.ssh/mindmate-vm_key.pem azureuser@<PUBLIC_IP>
```

**Install system deps:**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv nginx cmake build-essential git rsync
```

**Create venv:**
```bash
python3 -m venv ~/venvs/mindmate
source ~/venvs/mindmate/bin/activate
```

**Install Python packages (CPU-only):**
```bash
pip install llama-cpp-python --no-cache-dir
pip install fastapi "uvicorn[standard]" python-multipart
```

> For GPU offload (if VM has a GPU), use instead:
> `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir`

---

## Phase 4 — Upload code and model

Run these from your **local machine**:

```bash
# deploy/ folder
rsync -avz --progress \
  /Users/aryanjain/projects/mindmate/deploy/ \
  azureuser@<PUBLIC_IP>:~/mindmate/deploy/

# GGUF model
rsync -avz --progress \
  /Users/aryanjain/projects/mindmate/exports/mindmate_llama_sft_ck1600/ \
  azureuser@<PUBLIC_IP>:~/mindmate/exports/mindmate_llama_sft_ck1600/

# System prompt
rsync -avz \
  /Users/aryanjain/projects/mindmate/inference/system_prompt.txt \
  azureuser@<PUBLIC_IP>:~/mindmate/inference/system_prompt.txt
```

**On the server — activate venv in start.sh:**
```bash
sed -i 's|# source ~/venvs/mindmate/bin/activate|source ~/venvs/mindmate/bin/activate|' \
  ~/mindmate/deploy/start.sh
chmod +x ~/mindmate/deploy/start.sh
```

---

## Phase 5 — systemd service

```bash
sudo nano /etc/systemd/system/mindmate.service
```

Paste:
```ini
[Unit]
Description=MindMate / Anchor FastAPI server
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/mindmate/deploy
ExecStart=/home/azureuser/mindmate/deploy/start.sh
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable mindmate
sudo systemctl start mindmate
sudo systemctl status mindmate    # should show "active (running)"

# Tail logs
sudo journalctl -u mindmate -f
```

---

## Phase 6 — Nginx reverse proxy

```bash
sudo nano /etc/nginx/sites-available/mindmate
```

Paste:
```nginx
server {
    listen 80;
    server_name <PUBLIC_IP>;

    client_max_body_size 1M;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Required for SSE streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
        add_header X-Accel-Buffering no;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/mindmate /etc/nginx/sites-enabled/
sudo nginx -t              # must print "syntax is ok"
sudo systemctl restart nginx
```

---

## Phase 7 — Verify

Open `http://<PUBLIC_IP>` in a browser. You should see the Anchor chat UI.

Send a message and confirm token streaming works. Check server logs:
```bash
sudo journalctl -u mindmate -f
ls ~/mindmate/deploy/logs/
```

---

## Optional — SSL / HTTPS

Only needed if you attach a custom domain. Point the domain's A record to the public IP, then:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## Cost management

| Resource | Cost |
|---|---|
| Standard_D4as_v5 (on-demand) | ~$0.19/hr (~$140/mo running 24/7) |
| Azure for Students credit | $100 free — covers ~22 days |
| 32 GB Standard SSD | ~$2/mo |

**Stop the VM when not in use** to stretch credits:
```bash
az vm deallocate -g mindmate-rg -n mindmate-vm   # stop + deallocate (no compute charge)
az vm start      -g mindmate-rg -n mindmate-vm   # restart
```

---

## Re-deploying code changes

After editing files locally, re-sync and restart:
```bash
rsync -avz --progress \
  /Users/aryanjain/projects/mindmate/deploy/ \
  azureuser@<PUBLIC_IP>:~/mindmate/deploy/

ssh -i ~/.ssh/mindmate-vm_key.pem azureuser@<PUBLIC_IP> \
  "sudo systemctl restart mindmate"
```
