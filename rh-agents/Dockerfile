FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg build-essential git pkg-config openssh-server wget \
  && rm -rf /var/lib/apt/lists/*

# Node.js 22 (via NodeSource) + enable Corepack
RUN mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
  && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
  && apt-get update && apt-get install -y --no-install-recommends nodejs \
  && corepack enable \
  && rm -rf /var/lib/apt/lists/*

# SSH setup
RUN mkdir -p /var/run/sshd \
  && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
  && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
  && sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
  && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
  && sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config \
  && sed -i 's/session\s*required\s*pam_loginuid.so/session optional pam_loginuid.so/g' /etc/pam.d/sshd \
  && mkdir -p /root/.ssh && chmod 700 /root/.ssh

# (Optional) Show versions at build time
RUN python --version && node --version && npm --version

RUN npm install -g @google/gemini-cli@latest

#COPY requirements.txt requirements_dev.txt /app/
#RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir -r requirements_dev.txt

COPY . .

# Use Docker BuildKit secrets to securely inject the SSH public key
# This prevents the key from appearing in 'docker history'
RUN --mount=type=secret,id=ssh_pub_key \
    if [ -f /run/secrets/ssh_pub_key ]; then \
      cat /run/secrets/ssh_pub_key > /root/.ssh/authorized_keys && \
      chmod 600 /root/.ssh/authorized_keys; \
    fi

EXPOSE 22

# Start SSH daemon in foreground
CMD ["/usr/sbin/sshd", "-D", "-e"]