import asyncio
import io
import os
import random
import re
import time
import cairosvg
import ollama
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI

load_dotenv()

RWTH_API_KEY = os.getenv("RWTH_API_KEY", "")
rwth_client = AsyncOpenAI(
    base_url="https://chat.kiconnect.nrw/api/v1",
    api_key=RWTH_API_KEY if RWTH_API_KEY else "dummy-key",
    timeout=90.0,
    max_retries=2
)

app = FastAPI()

DEFAULT_DRAWER = "rwth:gpt-oss-120b"
GUESSER_MODEL = "qwen2.5vl:3b"

WORD_BANK = [
    "apple", "clock", "house", "car", "guitar", "bicycle", "airplane",
    "candle", "cactus", "chair", "umbrella", "bridge", "robot", "spider",
    "tree", "ladder", "camera", "boat", "cloud", "pizza", "sword", "snowman",
    "banana", "sun", "flower", "bottle", "glasses", "scissors"
]

connected_players: dict[WebSocket, dict] = {}

game_state = {
    "match_running": False,
    "is_paused": False,
    "is_round_active": False,
    "active_task": None,       # Stores active asyncio.Task to prevent ghost loops
    "word": "",
    "drawer": DEFAULT_DRAWER,
    "can_guess": False,
    "round_num": 0,
    "ai_score": 0,
    "ai_solved": False,
    "revealed_indices": set(),
    "time_left": 60,
    "current_svg": ""
}

def print_section(title: str, char="="):
    print(f"\n{char * 65}\n  {title}\n{char * 65}")

def extract_thinking_and_content(response: dict) -> tuple[str, str]:
    msg = response.get('message', {})
    content = msg.get('content', '') or ''
    thinking = msg.get('thinking', '') or ''

    if '<think>' in content:
        match = re.search(r'<think>([\s\S]*?)(?:<\/think>|$)', content)
        if match:
            thinking = match.group(1).strip()
            content = re.sub(r'<think>[\s\S]*?<\/think>', '', content).strip()

    return thinking.strip(), content.strip()

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]

def is_close_guess(guess: str, target: str) -> bool:
    dist = levenshtein_distance(guess, target)
    if dist == 1 and len(target) >= 3:
        return True
    if dist == 2 and len(target) >= 6:
        return True
    return False

def calculate_score(time_left: int, hints_revealed_count: int) -> int:
    score = (time_left * 10) - (hints_revealed_count * 100)
    return max(50, score)

def get_leaderboard_payload():
    roster = []
    for p in connected_players.values():
        roster.append({
            "name": p["name"],
            "score": p["score"],
            "solved": p["solved"],
            "is_ai": False
        })
    roster.append({
        "name": f"🤖 AI ({GUESSER_MODEL})",
        "score": game_state["ai_score"],
        "solved": game_state["ai_solved"],
        "is_ai": True
    })
    roster.sort(key=lambda x: x["score"], reverse=True)
    return roster

async def broadcast(message: dict):
    for ws in list(connected_players.keys()):
        try:
            await ws.send_json(message)
        except Exception:
            pass

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Pictionary Arena</title>
    <link rel="icon" href="data:,">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; margin: 0; height: 100vh; background: #0b0f19; color: #f8fafc; }
        
        #name-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center; z-index: 100; backdrop-filter: blur(4px); }
        .modal-box { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; width: 320px; text-align: center; }
        .modal-box input { width: 90%; padding: 10px; margin: 15px 0; border-radius: 6px; border: 1px solid #334155; background: #0b0f19; color: #fff; font-size: 15px; text-align: center; }

        #canvas-panel { flex: 2; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; border-right: 1px solid #1e293b; padding: 15px; overflow-y: auto; }
        #chat-panel { flex: 1; display: flex; flex-direction: column; background: #070a10; }
        
        #game-header { width: 380px; margin-bottom: 8px; display: flex; flex-direction: column; gap: 6px; }
        .status-row { display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 15px; }
        #timer-display { font-variant-numeric: tabular-nums; font-size: 18px; color: #38bdf8; }
        #word-blanks { font-family: monospace; font-size: 22px; letter-spacing: 5px; color: #fbbf24; font-weight: bold; }
        #progress-container { width: 100%; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
        #progress-bar { width: 100%; height: 100%; background: #38bdf8; transition: width 1s linear, background-color 0.5s; }

        #leaderboard-card { width: 380px; background: #111827; border: 1px solid #1e293b; border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
        .lb-header { background: #1e293b; padding: 6px 12px; font-size: 12px; font-weight: 700; color: #94a3b8; display: flex; justify-content: space-between; }
        .lb-list { display: flex; flex-direction: column; max-height: 90px; overflow-y: auto; }
        .lb-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 12px; border-bottom: 1px solid #1e293b; font-size: 13px; }
        .lb-name { display: flex; align-items: center; gap: 6px; font-weight: 600; }
        .badge-solved { color: #86efac; font-size: 12px; font-weight: bold; }

        #canvas-container { background: #ffffff; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); overflow: hidden; width: 380px; height: 380px; display: flex; align-items: center; justify-content: center; }
        #messages { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 8px; font-size: 14px; }
        .msg { padding: 8px 12px; border-radius: 6px; max-width: 85%; }
        .human { background: #2563eb; align-self: flex-end; }
        .peer { background: #1e293b; border: 1px solid #334155; align-self: flex-start; }
        .ai { background: #1e293b; border: 1px solid #334155; align-self: flex-start; }
        .ai-warn { background: #2d2218; border: 1px solid #78350f; color: #fbbf24; align-self: flex-start; font-size: 12px; }
        .system { color: #94a3b8; font-style: italic; align-self: center; font-size: 13px; }
        .hint { color: #fbbf24; font-weight: 600; align-self: center; font-size: 13px; }
        .close { background: #b45309; color: #fef3c7; font-weight: 600; align-self: flex-end; font-size: 13px; }
        .win-line { background: #15803d; color: #ffffff; font-weight: bold; align-self: center; text-align: center; width: 90%; }
        .intermission-line { background: #312e81; border: 1px solid #4f46e5; color: #e0e7ff; font-weight: 600; align-self: center; text-align: center; width: 90%; }
        .pause-banner { background: #78350f; border: 1px solid #f59e0b; color: #fef3c7; font-weight: 700; align-self: center; text-align: center; width: 90%; }

        #input-box { display: flex; padding: 12px; border-top: 1px solid #1e293b; background: #070a10; }
        #guess-input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #fff; font-size: 14px; outline: none; }
        #guess-input:disabled { background: #182030; color: #64748b; cursor: not-allowed; }
        
        #controls { margin-top: 10px; display: flex; gap: 8px; width: 380px; justify-content: center; align-items: center; }
        select { padding: 8px 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #fff; font-size: 13px; }
        button { padding: 8px 12px; border-radius: 6px; border: none; color: #fff; font-weight: 600; cursor: pointer; transition: opacity 0.15s; font-size: 13px; }
        button:hover { opacity: 0.9; }
        button.start-btn { background: #059669; }
        button.pause-btn { background: #d97706; }
        button.stop-btn { background: #dc2626; }
        button.inspector-btn { background: #475569; }

        #inspector-container { width: 380px; margin-top: 10px; display: none; flex-direction: column; background: #050811; border: 1px solid #1e293b; border-radius: 8px; font-family: monospace; font-size: 11px; }
        .insp-tab-header { display: flex; background: #0d1322; border-bottom: 1px solid #1e293b; }
        .insp-tab { flex: 1; padding: 6px; text-align: center; cursor: pointer; color: #94a3b8; font-weight: bold; }
        .insp-tab.active { background: #1e293b; color: #38bdf8; }
        .insp-body { padding: 8px; max-height: 180px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; color: #cbd5e1; }
    </style>
</head>
<body>
    <div id="name-modal">
        <div class="modal-box">
            <h2>Join AI Pictionary</h2>
            <p style="color: #94a3b8; font-size: 13px;">Choose a nickname for the scoreboard</p>
            <input type="text" id="player-name-input" placeholder="Your name (e.g. Alex)" maxlength="16" onkeydown="if(event.key==='Enter') joinGame()"/>
            <button class="start-btn" style="width: 95%;" onclick="joinGame()">Enter Arena</button>
        </div>
    </div>

    <div id="canvas-panel">
        <div id="leaderboard-card">
            <div class="lb-header">
                <span>RANKED LEADERBOARD</span>
                <span id="round-tag">LOBBY</span>
            </div>
            <div class="lb-list" id="lb-rows"></div>
        </div>

        <div id="game-header">
            <div class="status-row">
                <span id="word-blanks">_ _ _ _ _</span>
                <span id="timer-display">⏳ 60s</span>
            </div>
            <div id="progress-container">
                <div id="progress-bar"></div>
            </div>
        </div>

        <div id="canvas-container">
            <svg id="live-svg" viewBox="0 0 300 300" width="380" height="380" xmlns="http://www.w3.org/2000/svg">
                <style>
                    * { stroke-linecap: round; stroke-linejoin: round; }
                    rect.canvas-bg { fill: #ffffff !important; stroke: none !important; }
                </style>
                <rect class="canvas-bg" width="300" height="300"/>
            </svg>
        </div>

        <div id="controls">
            <button id="start-btn" class="start-btn" onclick="startMatch()">▶ Start Game</button>
            <button id="pause-btn" class="pause-btn" style="display:none;" onclick="togglePause()">⏸ Pause</button>
            <button id="stop-btn" class="stop-btn" style="display:none;" onclick="stopMatch()">⏹ End</button>
            <button class="inspector-btn" onclick="toggleInspector()">🔬 Inspector</button>
            <select id="drawer-select" onchange="updateDrawer()">
                <!-- RWTH High-Capacity Cloud Models -->
                <option value="rwth:gpt-oss-120b" selected>RWTH: gpt-oss-120b (120B Remote)</option>
                <option value="rwth:mistralai-mistral-small-4-119b">RWTH: Mistral Small (119B Remote)</option>
                <option value="rwth:Qwen 3.8 27B">RWTH: Qwen 3.8 27B (Vision + Reasoning)</option>

                <!-- Commercial Cloud Fallbacks (Uses Quota) -->
                <option value="rwth:gpt-5.5">RWTH: GPT-5.5 (Flagship)</option>
                <option value="rwth:gpt-5.4-mini">RWTH: GPT-5.4 Mini</option>
                
                <!-- Local WSL Ollama Fallbacks -->
                <option value="gemma4:e4b">Local: gemma4:e4b (4B)</option>
                <option value="qwen3.5:2b">Local: qwen3.5:2b (2B)</option>
                <option value="qwen2.5:3b">Local: qwen2.5:3b (3B)</option>
            </select>
        </div>

        <div id="inspector-container">
            <div class="insp-tab-header">
                <div class="insp-tab active" id="tab-btn-drawer" onclick="switchInspTab('drawer')">🧠 Drawer Thought & SVG</div>
                <div class="insp-tab" id="tab-btn-guesser" onclick="switchInspTab('guesser')">👁️ Guesser Evaluation</div>
            </div>
            <div class="insp-body" id="insp-content-drawer">Awaiting drawer output...</div>
            <div class="insp-body" id="insp-content-guesser" style="display:none;">Awaiting guesser cycles...</div>
        </div>
    </div>

    <div id="chat-panel">
        <div id="messages"></div>
        <div id="input-box">
            <input type="text" id="guess-input" disabled placeholder="Waiting for match to start..." onkeydown="if(event.key==='Enter') sendGuess()"/>
        </div>
    </div>

    <script>
        let ws = null;
        let myName = "";
        let matchRunning = false;
        let isPaused = false;
        let hasSolvedCurrentRound = false;

        const svgContainer = document.getElementById("live-svg");
        const messages = document.getElementById("messages");
        const timerDisplay = document.getElementById("timer-display");
        const progressBar = document.getElementById("progress-bar");
        const wordBlanks = document.getElementById("word-blanks");
        const guessInput = document.getElementById("guess-input");
        const lbRows = document.getElementById("lb-rows");
        const roundTag = document.getElementById("round-tag");

        const startBtn = document.getElementById("start-btn");
        const pauseBtn = document.getElementById("pause-btn");
        const stopBtn = document.getElementById("stop-btn");

        const inspContainer = document.getElementById("inspector-container");
        const inspDrawer = document.getElementById("insp-content-drawer");
        const inspGuesser = document.getElementById("insp-content-guesser");

        const svgFrame = `<style>
            * { stroke-linecap: round; stroke-linejoin: round; }
            rect.canvas-bg { fill: #ffffff !important; stroke: none !important; }
        </style><rect class="canvas-bg" width="300" height="300"/>`;

        function joinGame() {
            const val = document.getElementById("player-name-input").value.trim();
            if (!val) return;
            myName = val;
            document.getElementById("name-modal").style.display = "none";
            initWebSocket();
            // Sync whatever model is currently showing in the dropdown immediately
            updateDrawer();
        }

        function toggleInspector() {
            inspContainer.style.display = (inspContainer.style.display === "flex") ? "none" : "flex";
        }

        function switchInspTab(tab) {
            if (tab === 'drawer') {
                document.getElementById("tab-btn-drawer").className = "insp-tab active";
                document.getElementById("tab-btn-guesser").className = "insp-tab";
                inspDrawer.style.display = "block";
                inspGuesser.style.display = "none";
            } else {
                document.getElementById("tab-btn-drawer").className = "insp-tab";
                document.getElementById("tab-btn-guesser").className = "insp-tab active";
                inspDrawer.style.display = "none";
                inspGuesser.style.display = "block";
            }
        }

        function initWebSocket() {
            const protocol = location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${location.host}/ws`);

            ws.onopen = () => {
                ws.send(JSON.stringify({ type: "register", name: myName }));
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === "stroke_update") {
                    svgContainer.innerHTML = svgFrame + data.svg;
                } else if (data.type === "chat") {
                    addMessage(data.text, data.sender);
                } else if (data.type === "prep_round") {
                    isPaused = false;               // 1. Resets pause state so input unlocks on stroke 1
                    hasSolvedCurrentRound = false;  // 2. Resets solve status for the new round
                    wordBlanks.innerText = data.blanks;
                    svgContainer.innerHTML = svgFrame;
                    messages.innerHTML = '';
                    roundTag.innerText = `ROUND ${data.round}`;
                    guessInput.disabled = true;
                    guessInput.placeholder = "🎨 Drawer is sketching... Locked!";
                    addMessage(`--- Round ${data.round} Started! Object has ${data.length} letters. ---`, 'system');
                    inspDrawer.innerText = "Generating drawing plan...";     // 3. Clears old drawing plan
                    inspGuesser.innerText = "No guesses yet this round.";   // 4. Clears old guess evaluation logs
                } else if (data.type === "round_active") {
                    if (!isPaused && !hasSolvedCurrentRound) {
                        guessInput.disabled = false;
                        guessInput.placeholder = "Type your guess here...";
                        guessInput.focus();
                    }
                } else if (data.type === "midway_sync") {
                    svgContainer.innerHTML = svgFrame + data.svg;
                    wordBlanks.innerText = data.blanks;
                    roundTag.innerText = data.is_paused ? "PAUSED" : `ROUND ${data.round}`;
                    isPaused = data.is_paused;
                    if (data.can_guess && !isPaused) {
                        guessInput.disabled = false;
                        guessInput.placeholder = "Type your guess here...";
                    }
                    addMessage(`Joined mid-round! Guessing the ${data.length}-letter object.`, 'system');
                } else if (data.type === "timer_tick") {
                    timerDisplay.innerText = `⏳ ${data.time_left}s`;
                    const pct = (data.time_left / 60) * 100;
                    progressBar.style.width = pct + "%";
                    if (pct > 50) progressBar.style.backgroundColor = "#38bdf8";
                    else if (pct > 25) progressBar.style.backgroundColor = "#fbbf24";
                    else progressBar.style.backgroundColor = "#ef4444";
                } else if (data.type === "hint_update") {
                    wordBlanks.innerText = data.blanks;
                    addMessage(`Hint revealed: ${data.blanks}`, 'hint');
                } else if (data.type === "lock_input") {
                    hasSolvedCurrentRound = true;
                    guessInput.disabled = true;
                    guessInput.placeholder = "You solved it! Waiting for round to finish...";
                } else if (data.type === "leaderboard") {
                    renderLeaderboard(data.roster);
                } else if (data.type === "inspector_log") {
                    if (data.channel === "drawer") {
                        inspDrawer.innerText = `=== MODEL: ${data.model} ===\\n[THINKING]:\\n${data.thinking || '(None / Direct generation)'}\\n\\n[RAW SVG]:\\n${data.raw}\\n\\n[EXTRACTED STROKES]: ${data.stroke_count}`;
                    } else if (data.channel === "guesser") {
                        const prev = inspGuesser.innerText === "No guesses yet this round." ? "" : inspGuesser.innerText + "\\n\\n";
                        inspGuesser.innerText = prev + `[t=${data.time}s | Hint: ${data.pattern}]\\nCandidates: ${data.candidates}\\nThinking: ${data.thinking || '(None)'}\\nRaw Guess: '${data.raw_guess}' -> Sanitized: '${data.clean_guess}'\\nVerdict: ${data.verdict}`;
                        inspGuesser.scrollTop = inspGuesser.scrollHeight;
                    }
                } else if (data.type === "intermission") {
                    wordBlanks.innerText = data.revealed_word.toUpperCase();
                    timerDisplay.innerText = `⏳ ${data.countdown}s`;
                    progressBar.style.width = "0%";
                    guessInput.disabled = true;
                    if (data.countdown === 5) {
                        addMessage(`Word was '${data.revealed_word}'. Next round starting in 5s...`, 'intermission-line');
                    }
                } else if (data.type === "pause_state") {
                    isPaused = data.paused;
                    if (isPaused) {
                        pauseBtn.innerText = "▶ Resume";
                        pauseBtn.className = "start-btn";
                        roundTag.innerText = "PAUSED";
                        guessInput.disabled = true;
                        guessInput.placeholder = "⏸ Match is paused by host...";
                        addMessage("Match has been paused.", "pause-banner");
                    } else {
                        pauseBtn.innerText = "⏸ Pause";
                        pauseBtn.className = "pause-btn";
                        roundTag.innerText = `ROUND ${data.round}`;
                        if (!hasSolvedCurrentRound) {
                            guessInput.disabled = false;
                            guessInput.placeholder = "Type your guess here...";
                            guessInput.focus();
                        }
                        addMessage("Match resumed!", "system");
                    }
                } else if (data.type === "match_state") {
                    matchRunning = data.running;
                    if (matchRunning) {
                        startBtn.style.display = "none";
                        pauseBtn.style.display = "inline-block";
                        stopBtn.style.display = "inline-block";
                    } else {
                        startBtn.style.display = "inline-block";
                        pauseBtn.style.display = "none";
                        stopBtn.style.display = "none";
                        roundTag.innerText = "LOBBY";
                        guessInput.disabled = true;
                        guessInput.placeholder = "Match ended. Click Start Game to begin.";
                    }
                }
            };
        }

        function renderLeaderboard(roster) {
            lbRows.innerHTML = "";
            const medals = ["🥇", "🥈", "🥉"];
            roster.forEach((p, idx) => {
                const row = document.createElement("div");
                row.className = "lb-row";
                const rankPrefix = medals[idx] || `${idx + 1}.`;
                row.innerHTML = `
                    <div class="lb-name">
                        <span>${rankPrefix}</span>
                        <span>${p.name} ${p.name === myName ? '(You)' : ''}</span>
                        ${p.solved ? '<span class="badge-solved">✓</span>' : ''}
                    </div>
                    <span style="font-weight: 700; color: #38bdf8;">${p.score} pts</span>
                `;
                lbRows.appendChild(row);
            });
        }

        function addMessage(text, type) {
            const div = document.createElement("div");
            div.className = `msg ${type}`;
            div.innerText = text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        function sendGuess() {
            if (guessInput.value.trim() && !guessInput.disabled && !isPaused) {
                ws.send(JSON.stringify({ type: "guess", text: guessInput.value.trim() }));
                guessInput.value = "";
            }
        }

        function startMatch() {
            isPaused = false;  // Reset pause state on new match
            const activeDrawer = document.getElementById("drawer-select").value;
            fetch(`/match/start?drawer=${encodeURIComponent(activeDrawer)}`);
        }

        function togglePause() {
            if (!isPaused) {
                fetch(`/match/pause`);
            } else {
                fetch(`/match/resume`);
            }
        }

        function stopMatch() {
            isPaused = false;  // Reset pause state on stop
            fetch(`/match/stop`);
        }

        function updateDrawer() {
            const d = document.getElementById("drawer-select").value;
            fetch(`/drawer/set?model=${encodeURIComponent(d)}`);
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get_ui():
    return HTMLResponse(HTML_UI)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_players[websocket] = {"name": "Guest", "score": 0, "solved": False}
    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("type")

            if event_type == "register":
                raw_name = data.get("name", "Player").strip()
                connected_players[websocket]["name"] = raw_name[:16] if raw_name else "Player"
                
                await broadcast({"type": "leaderboard", "roster": get_leaderboard_payload()})
                await websocket.send_json({"type": "match_state", "running": game_state["match_running"]})

                if game_state["is_round_active"]:
                    blanks = build_hint_pattern(game_state["word"], game_state["revealed_indices"])
                    await websocket.send_json({
                        "type": "midway_sync",
                        "round": game_state["round_num"],
                        "length": len(game_state["word"]),
                        "blanks": blanks,
                        "svg": game_state["current_svg"],
                        "can_guess": game_state["can_guess"],
                        "is_paused": game_state["is_paused"]
                    })

            elif event_type == "guess":
                if not game_state["is_round_active"] or not game_state["can_guess"] or game_state["is_paused"]:
                    continue

                player = connected_players.get(websocket)
                if not player or player["solved"]:
                    continue

                guess = data.get("text", "").strip().lower()
                target = game_state["word"].strip().lower()

                if guess == target:
                    player["solved"] = True
                    pts = calculate_score(game_state["time_left"], len(game_state["revealed_indices"]))
                    player["score"] += pts

                    await websocket.send_json({"type": "lock_input"})
                    await broadcast({
                        "type": "chat",
                        "sender": "win-line",
                        "text": f"🎉 {player['name']} guessed the word! (+{pts} pts)"
                    })
                    await broadcast({"type": "leaderboard", "roster": get_leaderboard_payload()})

                    all_humans_done = all(p["solved"] for p in connected_players.values())
                    if all_humans_done and game_state["ai_solved"]:
                        game_state["is_round_active"] = False
                elif is_close_guess(guess, target):
                    await websocket.send_json({
                        "type": "chat",
                        "sender": "close",
                        "text": f"'{guess}' is very close!"
                    })
                else:
                    for client_ws, info in connected_players.items():
                        sender_class = "human" if client_ws == websocket else "peer"
                        await client_ws.send_json({
                            "type": "chat",
                            "sender": sender_class,
                            "text": f"{player['name']}: {guess}"
                        })

    except WebSocketDisconnect:
        if websocket in connected_players:
            del connected_players[websocket]
            await broadcast({"type": "leaderboard", "roster": get_leaderboard_payload()})

def sanitize_and_parse_strokes(raw_svg: str) -> list[str]:
    svg_block = re.search(r"<svg[\s\S]*?<\/svg>", raw_svg, re.IGNORECASE)
    content = svg_block.group(0) if svg_block else raw_svg

    stroke_pattern = r"(<(path|circle|rect|line|ellipse|polyline|polygon)\b[^>]*?(?:\/?>|>[\s\S]*?<\/\2>))"
    matches = [m[0] for m in re.findall(stroke_pattern, content, re.IGNORECASE)]
    
    clean = []
    for s in matches:
        if 'width="300"' in s and 'height="300"' in s:
            continue
        if not s.endswith("/>") and not re.search(r"<\/\w+>$", s):
            s = s.rstrip(">") + "/>"

        if "fill=" not in s.lower():
            s = s.replace("/>", ' fill="none"/>', 1)
        else:
            s = re.sub(r'fill=["\']?(black|#000000|#000|#111111)["\']?', 'fill="none"', s, flags=re.IGNORECASE)

        if "stroke=" not in s.lower():
            s = s.replace("/>", ' stroke="#334155" stroke-width="4"/>', 1)
        elif "stroke-width=" not in s.lower():
            s = s.replace("/>", ' stroke-width="4"/>', 1)

        clean.append(s)
    return clean

def build_hint_pattern(word: str, revealed_indices: set[int]) -> str:
    return " ".join([ch.upper() if idx in revealed_indices else "_" for idx, ch in enumerate(word)])

def matches_pattern(guess: str, word: str, revealed_indices: set[int]) -> bool:
    if len(guess) != len(word):
        return False
    for idx in revealed_indices:
        if guess[idx] != word[idx]:
            return False
    return True

def get_candidate_words(word: str, revealed_indices: set[int]) -> list[str]:
    candidates = [w for w in WORD_BANK if matches_pattern(w, word, revealed_indices)]
    if word not in candidates:
        candidates.append(word)
    random.shuffle(candidates)
    return candidates[:8]

@app.get("/drawer/set")
async def set_drawer(model: str):
    game_state["drawer"] = model
    print(f"\n[CONFIG] Drawer model changed to: {model}")
    return {"status": "drawer updated", "drawer": model}

@app.get("/match/start")
async def start_match(drawer: str = None):
    if drawer:
        game_state["drawer"] = drawer
        print(f"\n[CONFIG] Match starting with drawer: {drawer}")

    if game_state["active_task"] and not game_state["active_task"].done():
        game_state["active_task"].cancel()
        try:
            await game_state["active_task"]
        except asyncio.CancelledError:
            pass

    game_state["match_running"] = True
    game_state["is_paused"] = False
    game_state["active_task"] = asyncio.create_task(run_continuous_match_loop())
    return {"status": "match started", "drawer": game_state["drawer"]}

@app.get("/match/pause")
async def pause_match():
    if game_state["match_running"]:
        game_state["is_paused"] = True
        print(f"\n[MATCH PAUSED] Host paused Round {game_state['round_num']}")
        await broadcast({"type": "pause_state", "paused": True, "round": game_state["round_num"]})
    return {"status": "paused"}

@app.get("/match/resume")
async def resume_match():
    if game_state["match_running"]:
        game_state["is_paused"] = False
        print(f"\n[MATCH RESUMED] Host resumed Round {game_state['round_num']}")
        await broadcast({"type": "pause_state", "paused": False, "round": game_state["round_num"]})
    return {"status": "resumed"}

@app.get("/match/stop")
async def stop_match():
    game_state["match_running"] = False
    game_state["is_paused"] = False
    game_state["is_round_active"] = False
    if game_state["active_task"] and not game_state["active_task"].done():
        game_state["active_task"].cancel()
    print(f"\n[MATCH STOPPED] Match ended by host.")
    await broadcast({"type": "match_state", "running": False})
    return {"status": "match stopping"}

async def run_continuous_match_loop():
    await broadcast({"type": "match_state", "running": True})

    try:
        while game_state["match_running"]:
            target_word = random.choice(WORD_BANK)
            await run_single_round(target_word, game_state["drawer"])

            if not game_state["match_running"]:
                break

            # 5-second intermission
            for count in range(5, 0, -1):
                while game_state["is_paused"] and game_state["match_running"]:
                    await asyncio.sleep(0.5)
                if not game_state["match_running"]:
                    break
                await broadcast({
                    "type": "intermission",
                    "countdown": count,
                    "revealed_word": target_word
                })
                await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        print("\n[MATCH LOOP CANCELLED CLEANLY]")
    finally:
        await broadcast({"type": "match_state", "running": False})

async def run_single_round(secret_word: str, drawer_model: str):
    current_round_id = game_state["round_num"] + 1
    game_state["round_num"] = current_round_id
    game_state["word"] = secret_word
    game_state["is_round_active"] = True
    game_state["can_guess"] = False
    game_state["ai_solved"] = False
    game_state["revealed_indices"] = set()
    game_state["time_left"] = 60
    game_state["current_svg"] = ""

    for p in connected_players.values():
        p["solved"] = False

    attempted_ai_guesses = set()
    blanks = build_hint_pattern(secret_word, game_state["revealed_indices"])

    print_section(f"ROUND {current_round_id} INITIATED | TARGET: '{secret_word.upper()}' ({len(secret_word)} letters)")
    print(f"🎨 Drawer Model : {drawer_model}")
    print(f"👁️ Guesser Model: {GUESSER_MODEL}")

    await broadcast({
        "type": "prep_round",
        "round": current_round_id,
        "length": len(secret_word),
        "blanks": blanks
    })
    await broadcast({"type": "leaderboard", "roster": get_leaderboard_payload()})
    await broadcast({"type": "chat", "sender": "system", "text": f"[{drawer_model}] is sketching..."})

    prompt = f"""You are playing Pictionary. Draw a highly detailed, progressive vector sketch of: '{secret_word}'.
Canvas: 300x300. Center is (150, 150).

DRAW IN 3 CHRONOLOGICAL PHASES (Output 12 to 16 total shapes):
- Phase 1 (Base & Background): Primary silhouette, background contours, ground lines, or large fills.
- Phase 2 (Surfaces & Colors): Main body volume, fills, branches, cushions, or rims.
- Phase 3 (Fine Details & Foreground): Spines/needles, wheels, buttons, texture lines, or toppings.
CRITICAL: Draw from background to foreground so later shapes do not cover earlier ones!

RULES FOR '{secret_word}':
1. Output 12 to 16 distinct geometric elements (<rect>, <circle>, <ellipse>, <path>, <line>).
2. Choose realistic colors matching '{secret_word}'. Use fill="..." for solid parts and stroke="..." for outlines.
3. Every shape must have stroke-width="3" or "4".
4. OUTPUT FORMAT:
   - First, inside <plan>...</plan>, briefly plan the layers, coordinates, and colors in 2 to 4 bullet points.
   - Then, output the raw <svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">...</svg> block.
   - Do NOT include any markdown commentary, greetings, or explanations outside these blocks."""

    drawer_thinking = ""
    svg_content = ""

    try:
        if drawer_model.startswith("rwth:"):
            actual_model_name = drawer_model.replace("rwth:", "")
            
            # Configure request parameters
            api_kwargs = {
                "model": actual_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 4000,  # Raised from 1500 to prevent token exhaustion
                "timeout": 90.0
            }

            # For GPT-5 models, keep reasoning light so it starts drawing immediately
            if "gpt-5" in actual_model_name:
                api_kwargs["extra_body"] = {"reasoning_effort": "low"}

            api_res = await rwth_client.chat.completions.create(**api_kwargs)
            choice = api_res.choices[0]
            raw_text = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", "")
            thinking_text = getattr(choice.message, "reasoning_content", "") or ""

            # If the model exhausted tokens during reasoning, trigger the fallback
            if finish_reason == "length" and not raw_text:
                raise RuntimeError(f"{actual_model_name} exhausted all tokens during reasoning before outputting SVG.")

            plan_match = re.search(r"<plan>([\s\S]*?)<\/plan>", raw_text, re.IGNORECASE)
            if plan_match:
                thinking_text = plan_match.group(1).strip()
                raw_text = re.sub(r"<plan>[\s\S]*?<\/plan>", "", raw_text, flags=re.IGNORECASE).strip()
            elif "<think>" in raw_text:
                think_match = re.search(r"<think>([\s\S]*?)<\/think>", raw_text, re.IGNORECASE)
                if think_match:
                    thinking_text = think_match.group(1).strip()
                    raw_text = re.sub(r"<think>[\s\S]*?<\/think>", "", raw_text, flags=re.IGNORECASE).strip()

            drawer_thinking, svg_content = thinking_text.strip(), raw_text.strip()
        else:
            response = await asyncio.to_thread(
                ollama.chat,
                model=drawer_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3}
            )
            drawer_thinking, svg_content = extract_thinking_and_content(response)

    except Exception as e:
        err_msg = f"Drawer ({drawer_model}) failed: {str(e)}"
        print(f"❌ [DRAWER ERROR]: {err_msg}")

        # If a cloud model timed out or threw an API error, fall back to offline local model
        if drawer_model.startswith("rwth:"):
            reason = "timed out" if "timeout" in str(e).lower() else "API error"
            print(f"⚠️ [FALLBACK]: Falling back to local gemma4:e4b ({reason}).")
            await broadcast({
                "type": "chat",
                "sender": "ai-warn",
                "text": f"RWTH model {reason}. Falling back to local gemma4:e4b..."
            })
            try:
                response = await asyncio.to_thread(
                    ollama.chat,
                    model="gemma4:e4b",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.3}
                )
                drawer_thinking, svg_content = extract_thinking_and_content(response)
                
                # UPDATE THE ACTIVE MODEL NAME HERE:
                drawer_model = "gemma4:e4b (fallback)"

            except Exception as local_err:
                print(f"❌ [LOCAL FALLBACK ERROR]: {local_err}")
                await broadcast({"type": "chat", "sender": "system", "text": "Drawer failed to generate shapes."})
                game_state["is_round_active"] = False
                return
        else:
            await broadcast({"type": "chat", "sender": "system", "text": "Drawer failed to generate shapes."})
            game_state["is_round_active"] = False
            return

    # Check if round was cancelled while waiting on model inference
    if not game_state["match_running"] or game_state["round_num"] != current_round_id:
        return

    print("\n" + "-" * 50)
    print(f"🧠 [DRAWER THINKING / REASONING ({drawer_model})]:")
    print(drawer_thinking if drawer_thinking else "(No separate thinking output generated)")
    print("-" * 50)
    print(f"🎨 [DRAWER RAW SVG ({drawer_model})]:")
    print(svg_content if svg_content else "(EMPTY CONTENT RETURNED)")
    print("-" * 50)

    strokes = sanitize_and_parse_strokes(svg_content)
    num_strokes = len(strokes)
    print(f"📦 [PARSED STROKES]: Extracted {num_strokes} valid geometric shapes\n")

    await broadcast({
        "type": "inspector_log",
        "channel": "drawer",
        "model": drawer_model,
        "thinking": drawer_thinking,
        "raw": svg_content,
        "stroke_count": num_strokes
    })

    if not strokes:
        await broadcast({"type": "chat", "sender": "system", "text": "Drawer failed to generate shapes."})
        game_state["is_round_active"] = False
        return

    DRAWING_WINDOW = 46.0
    stroke_schedule = [int(i * (DRAWING_WINDOW / num_strokes)) for i in range(num_strokes)]

    total_time = 60
    stroke_idx = 1
    game_state["current_svg"] = f"\n{strokes[0]}"
    strokes_finished_announced = False

    await broadcast({"type": "stroke_update", "svg": game_state["current_svg"]})
    game_state["can_guess"] = True
    await broadcast({"type": "round_active"})

    word_len = len(secret_word)

    for second_left in range(total_time, 0, -1):
        if not game_state["is_round_active"] or not game_state["match_running"] or game_state["round_num"] != current_round_id:
            break

        while game_state["is_paused"] and game_state["match_running"]:
            await asyncio.sleep(0.5)

        if not game_state["is_round_active"] or not game_state["match_running"] or game_state["round_num"] != current_round_id:
            break

        elapsed = total_time - second_left
        game_state["time_left"] = second_left
        await broadcast({"type": "timer_tick", "time_left": second_left})

        # Adaptive hints
        if word_len == 3 and second_left == 35 and len(game_state["revealed_indices"]) == 0:
            game_state["revealed_indices"].add(0)
            pattern = build_hint_pattern(secret_word, game_state["revealed_indices"])
            print(f"💡 [HINT REVEALED at t={second_left}s]: {pattern}")
            await broadcast({"type": "hint_update", "blanks": pattern})
        elif word_len in (4, 5):
            if second_left == 40 and len(game_state["revealed_indices"]) == 0:
                game_state["revealed_indices"].add(0)
                pattern = build_hint_pattern(secret_word, game_state["revealed_indices"])
                print(f"💡 [HINT REVEALED at t={second_left}s]: {pattern}")
                await broadcast({"type": "hint_update", "blanks": pattern})
            elif second_left == 20 and len(game_state["revealed_indices"]) == 1:
                avail = [i for i in range(1, word_len) if i not in game_state["revealed_indices"]]
                if avail:
                    game_state["revealed_indices"].add(random.choice(avail))
                    pattern = build_hint_pattern(secret_word, game_state["revealed_indices"])
                    print(f"💡 [HINT REVEALED at t={second_left}s]: {pattern}")
                    await broadcast({"type": "hint_update", "blanks": pattern})
        elif word_len >= 6:
            if second_left == 45 and len(game_state["revealed_indices"]) == 0:
                game_state["revealed_indices"].add(0)
                pattern = build_hint_pattern(secret_word, game_state["revealed_indices"])
                print(f"💡 [HINT REVEALED at t={second_left}s]: {pattern}")
                await broadcast({"type": "hint_update", "blanks": pattern})
            elif second_left == 25 and len(game_state["revealed_indices"]) == 1:
                avail = [i for i in range(1, word_len) if i not in game_state["revealed_indices"]]
                if avail:
                    game_state["revealed_indices"].add(random.choice(avail))
                    pattern = build_hint_pattern(secret_word, game_state["revealed_indices"])
                    print(f"💡 [HINT REVEALED at t={second_left}s]: {pattern}")
                    await broadcast({"type": "hint_update", "blanks": pattern})

        # Progressive stroke emission
        while stroke_idx < num_strokes and elapsed >= stroke_schedule[stroke_idx]:
            game_state["current_svg"] += f"\n{strokes[stroke_idx]}"
            await broadcast({"type": "stroke_update", "svg": game_state["current_svg"]})
            stroke_idx += 1
            if stroke_idx == num_strokes and not strokes_finished_announced:
                strokes_finished_announced = True
                print("🎨 [DRAWER COMPLETED]: All planned strokes rendered.")
                await broadcast({"type": "chat", "sender": "system", "text": "🎨 Sketch complete! Keep guessing until time runs out!"})

        # Guesser prediction cycle every 4 seconds
        if not game_state["ai_solved"] and elapsed % 4 == 0 and game_state["current_svg"]:
            full_svg = f"""<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
                <style>
                    * {{ stroke-linecap: round; stroke-linejoin: round; }}
                    rect.canvas-bg {{ fill: white !important; stroke: none !important; }}
                </style>
                <rect class="canvas-bg" width="300" height="300"/>
                {game_state['current_svg']}
            </svg>"""

            # png_bytes = cairosvg.svg2png(bytestring=full_svg.encode("utf-8"), output_width=300, output_height=300)
            png_bytes = await asyncio.to_thread(
                cairosvg.svg2png,
                bytestring=full_svg.encode("utf-8"),
                output_width=300,
                output_height=300
            )
            current_pattern = build_hint_pattern(secret_word, game_state["revealed_indices"])

            if game_state["revealed_indices"]:
                candidates = get_candidate_words(secret_word, game_state["revealed_indices"])
                candidate_str = ", ".join(candidates)
                guess_prompt = f"""Identify the object sketched in this image.
HINTS:
- Pattern: {current_pattern} ({len(secret_word)} letters)
- Options matching this pattern: [{candidate_str}]
Choose the SINGLE word from the list that best matches the sketch. Answer with ONLY that single word."""
            else:
                candidates = []
                candidate_str = "None (Open-ended)"
                guess_prompt = f"What simple object is sketched in this image? The word has {len(secret_word)} letters. Answer with ONLY the single lowercase noun."

            ai_resp = await asyncio.to_thread(
                ollama.chat,
                model=GUESSER_MODEL,
                messages=[{"role": "user", "content": guess_prompt, "images": [png_bytes]}],
                options={"temperature": 0.2}
            )

            guesser_thinking, raw_guess_content = extract_thinking_and_content(ai_resp)
            ai_guess = re.sub(r"[^\w]", "", raw_guess_content).strip().lower()

            if matches_pattern(ai_guess, secret_word, game_state["revealed_indices"]):
                if ai_guess == secret_word.lower():
                    verdict = "✅ CORRECT! Won round points"
                else:
                    verdict = "❌ Valid pattern, but incorrect object"
            else:
                verdict = "⚠️ INVALID (Violates letter pattern or length)"

            print(f"👁️ [GUESSER CYCLE @ t={second_left}s]")
            print(f"   Pattern     : {current_pattern}")
            print(f"   Candidates  : [{candidate_str}]")
            if guesser_thinking:
                print(f"   Thinking    : {guesser_thinking}")
            print(f"   Raw Output  : '{raw_guess_content}' -> Parsed: '{ai_guess}'")
            print(f"   Verdict     : {verdict}")

            await broadcast({
                "type": "inspector_log",
                "channel": "guesser",
                "time": second_left,
                "pattern": current_pattern,
                "candidates": candidate_str,
                "thinking": guesser_thinking,
                "raw_guess": raw_guess_content,
                "clean_guess": ai_guess,
                "verdict": verdict
            })

            if matches_pattern(ai_guess, secret_word, game_state["revealed_indices"]):
                if ai_guess == secret_word.lower():
                    game_state["ai_solved"] = True
                    pts = calculate_score(second_left, len(game_state["revealed_indices"]))
                    game_state["ai_score"] += pts

                    await broadcast({
                        "type": "chat",
                        "sender": "win-line",
                        "text": f"🤖 [{GUESSER_MODEL}] guessed the word! (+{pts} pts)"
                    })
                    await broadcast({"type": "leaderboard", "roster": get_leaderboard_payload()})

                    all_humans_done = all(p["solved"] for p in connected_players.values())
                    if all_humans_done:
                        game_state["is_round_active"] = False
                else:
                    if ai_guess not in attempted_ai_guesses:
                        attempted_ai_guesses.add(ai_guess)
                        await broadcast({"type": "chat", "sender": "ai", "text": f"[{GUESSER_MODEL}] guessed: {ai_guess}"})
            else:
                if ai_guess and ai_guess not in attempted_ai_guesses:
                    attempted_ai_guesses.add(ai_guess)
                    await broadcast({"type": "chat", "sender": "ai-warn", "text": f"[{GUESSER_MODEL}] tried: {ai_guess} (violates hint)"})

        await asyncio.sleep(1.0)

    game_state["is_round_active"] = False
    game_state["can_guess"] = False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)