FROM mcr.microsoft.com/devcontainers/base:jammy

ARG PIXI_VERSION=v0.83.0
ARG NODE_VERSION=22.23.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep fd-find ca-certificates curl xz-utils \
    && ln -sf /usr/bin/fdfind /usr/local/bin/fd \
    && arch="$(uname -m)" \
    && case "$arch" in \
         x86_64) node_arch=x64 ;; \
         aarch64) node_arch=arm64 ;; \
         *) echo "unsupported architecture: $arch" && exit 1 ;; \
       esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm /tmp/node.tar.xz \
    && node --version \
    && npm --version \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L -o /usr/local/bin/pixi -fsSL --compressed "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    && chmod +x /usr/local/bin/pixi \
    && pixi info

# set some user and workdir settings to work nicely with vscode
USER vscode
WORKDIR /home/vscode

RUN echo 'eval "$(pixi completion -s bash)"' >> /home/vscode/.bashrc \
    && curl -fsSL https://pi.dev/install.sh | sh \
    && pi_bin="$(command -v pi || find "$HOME" /usr/local/bin -name pi -type f 2>/dev/null | head -n 1)" \
    && test -n "$pi_bin" \
    && echo "export PATH=\"$(dirname "$pi_bin"):\$PATH\"" >> "$HOME/.bashrc" \
    && "$pi_bin" install npm:@probabl/pi-skore \
    && "$pi_bin" --version
