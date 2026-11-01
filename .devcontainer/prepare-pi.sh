#!/usr/bin/env bash
# Use the Codespace OpenRouter key and the session's DeepSeek default.
# A saved login in ~/.pi/agent/auth.json overrides OPENROUTER_API_KEY.
set -euo pipefail

node << 'JS'
const fs = require("fs");
const os = require("os");
const path = require("path");

const agent = path.join(os.homedir(), ".pi", "agent");
fs.mkdirSync(agent, { recursive: true });

const authPath = path.join(agent, "auth.json");
if (fs.existsSync(authPath)) {
  const data = JSON.parse(fs.readFileSync(authPath, "utf8"));
  delete data.openrouter;
  if (Object.keys(data).length === 0) fs.rmSync(authPath);
  else fs.writeFileSync(authPath, JSON.stringify(data, null, 2) + "\n");
}

const settingsPath = path.join(agent, "settings.json");
const settings = fs.existsSync(settingsPath)
  ? JSON.parse(fs.readFileSync(settingsPath, "utf8"))
  : {};
settings.defaultProjectTrust = "always";
settings.defaultProvider = "openrouter";
settings.defaultModel = "deepseek/deepseek-v4.1-flash-1017";
settings.enabledModels = ["deepseek/deepseek-v4.1-flash-1017"];
fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + "\n");

const trustPath = path.join(agent, "trust.json");
const trust = fs.existsSync(trustPath)
  ? JSON.parse(fs.readFileSync(trustPath, "utf8"))
  : {};
trust[fs.realpathSync(process.cwd())] = true;
fs.writeFileSync(trustPath, JSON.stringify(trust, null, 2) + "\n");
JS
