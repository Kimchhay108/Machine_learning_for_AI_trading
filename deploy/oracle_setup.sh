#!/usr/bin/env bash
# ==============================================================================
# Oracle Cloud Always Free Ubuntu Deployment Script for XAUUSD AI Trading Platform
# ==============================================================================
set -e

echo "=== [1/5] Updating System Packages ==="
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg lsb-release git

echo "=== [2/5] Configuring Host Firewall for Oracle Cloud ==="
# Oracle Cloud Ubuntu images block inbound ports at the OS iptables level by default.
# Open ports 80, 443, 3000 (Frontend), and 8000 (Backend API).
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 3000 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT 2>/dev/null || true

# Persist iptables rules across reboots
if command -v netfilter-persistent >/dev/null 2>&1; then
    sudo netfilter-persistent save
fi

echo "=== [3/5] Installing Docker & Docker Compose ==="
if ! command -v docker >/dev/null 2>&1; then
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
fi

echo "=== [4/5] Building and Launching Docker Containers ==="
docker compose down || true
docker compose up -d --build

echo "=== [5/5] Verifying Deployment ==="
sleep 5
docker compose ps

SERVER_IP=$(curl -s https://api.ipify.org || echo "YOUR_SERVER_IP")
echo ""
echo "======================================================================"
echo " SUCCESS! XAUUSD AI Trading Platform is running on Oracle Cloud!"
echo " Web Dashboard:  http://${SERVER_IP}:3000"
echo " FastAPI Docs:   http://${SERVER_IP}:8000/docs"
echo "======================================================================"
