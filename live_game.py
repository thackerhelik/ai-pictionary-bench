import asyncio
import io
import random
import re
import time
import cairosvg
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse

app = FastAPI()

DEFAULT_DRAWER = "gemma4:e4b"
GUESSER_MODEL = "qwen2.5vl:3b"

WORD_BANK = [
    "apple", "clock", "house", "car", "guitar", "bicycle", "airplane",
    "candle", "cactus", "chair", "umbrella", "bridge", "robot", "spider",
    "tree", "ladder", "camera", "boat", "cloud", "pizza", "sword", "snowman",
    "banana", "sun", "flower", "bottle", "glasses", "scissors"
]

# Player Registry: { ws: {"id": str, "name": str, "score": int, "solved": bool} }
connected_players: dict[WebSocket, dict] = {}

game_state = {
    "word": "",
    "drawer": DEFAULT_DRAWER,
    "is_running": False,
    "can_guess": False,
    "round_num": 0,
    "ai_score": 0,
    "ai_solved": False,
    "revealed_indices": set(),
    "time_left": 60
}

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
        
        /* Nickname Overlay */
        #name-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.85); display: flex; align-items: center; justify-content: center; z-index: 100; backdrop-filter: blur(4px); }
        .modal-box { background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; width: 320px; text-align: center; }
        .modal-box input { width: 90%; padding: 10px; margin: 15px 0; border-radius: 6px; border: 1px solid #334155; background: #0b0f19; color: #fff; font-size: 15px; text-align: center; }

        #canvas-panel { flex: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; border-right: 1px solid #1e293b; padding: 20px; }
        #chat-panel { flex: 1; display: flex; flex-direction: column; background: #070a10; }
        
        /* Header & Leaderboard */
        #game-header { width: 380px; margin-bottom: 10px; display: flex; flex-direction: column; gap: 6px; }
        .status-row { display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 15px; }
        #timer-display { font-variant-numeric: tabular-nums; font-size: 18px; color: #38bdf8; }
        #word-blanks { font-family: monospace; font-size: 22px; letter-spacing: 5px; color: #fbbf24; font-weight: bold; }
        #progress-container { width: 100%; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
        #progress-bar { width: 100%; height: 100%; background: #38bdf8; transition: width 1s linear, background-color 0.5s; }

        /* Dynamic Ranked Table */
        #leaderboard-card { width: 380px; background: #111827; border: 1px solid #1e293b; border-radius: 8px; margin-bottom: 10px; overflow: hidden; }
        .lb-header { background: #1e293b; padding: 6px 12px; font-size: 12px; font-weight: 700; color: #94a3b8; display: flex; justify-content: space-between; }
        .lb-list { display: flex; flex-direction: column; max-height: 110px; overflow-y: auto; }
        .lb-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 12px; border-bottom: 1px solid #1e293b; font-size: 13px; }
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
        .round-end-banner { background: #1e1b4b; border: 1px solid #4338ca; color: #c7d2fe; font-weight: 600; align-self: center; text-align: center; width: 90%; }
        
        #input-box { display: flex; padding: 12px; border-top: 1px solid #1e293b; background: #070a10; }
        #guess-input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #fff; font-size: 14px; outline: none; }
        #guess-input:disabled { background: #182030; color: #64748b; cursor: not-allowed; }
        
        #controls { margin-top: 15px; display: flex; flex-direction: column; gap: 10px; align-items: center; width: 380px; }
        .control-row { display: flex; gap: 8px; width: 100%; justify-content: center; align-items: center; }
        select, input[type="text"] { padding: 8px 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #fff; font-size: 13px; }
        button { padding: 8px 14px; border-radius: 6px; border: none; background: #2563eb; color: #fff; font-weight: 600; cursor: pointer; transition: background 0.15s; }
        button:hover { background: #1d4ed8; }
        button.action-btn { background: #059669; }
        button.action-btn:hover { background: #047857; }
    </style>
</head>
<body>
    <div id="name-modal">
        <div class="modal-box">
            <h2>Join AI Pictionary</h2>
            <p style="color: #94a3b8; font-size: 13px;">Choose a display name for the scoreboard</p>
            <input type="text" id="player-name-input" placeholder="Your name (e.g. Alex)" maxlength="16" onkeydown="if(event.key==='Enter') joinGame()"/>
            <button class="action-btn" style="width: 95%;" onclick="joinGame()">Enter Arena</button>
        </div>
    </div>

    <div id="canvas-panel">
        <div id="leaderboard-card">
            <div class="lb-header">
                <span>RANKED LEADERBOARD</span>
                <span id="round-tag">ROUND 0</span>
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
            <div class="control-row">
                <button class="action-btn" onclick="startRandom()">🎲 Next Random Word</button>
                <select id="drawer-select">
                    <option value="gemma4:e4b">Drawer: gemma4:e4b</option>
                    <option value="qwen3.5:2b">Drawer: qwen3.5:2b</option>
                    <option value="qwen2.5:3b">Drawer: qwen2.5:3b</option>
                </select>
            </div>
            <div class="control-row">
                <input type="text" id="custom-word" placeholder="Or test custom word..." style="flex: 1;" />
                <button onclick="startCustom()">Draw</button>
            </div>
        </div>
    </div>

    <div id="chat-panel">
        <div id="messages"></div>
        <div id="input-box">
            <input type="text" id="guess-input" disabled placeholder="Waiting for round to begin..." onkeydown="if(event.key==='Enter') sendGuess()"/>
        </div>
    </div>

    <script>
        let ws = null;
        let myName = "";
        const svgContainer = document.getElementById("live-svg");
        const messages = document.getElementById("messages");
        const timerDisplay = document.getElementById("timer-display");
        const progressBar = document.getElementById("progress-bar");
        const wordBlanks = document.getElementById("word-blanks");
        const guessInput = document.getElementById("guess-input");
        const lbRows = document.getElementById("lb-rows");
        const roundTag = document.getElementById("round-tag");

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
                    wordBlanks.innerText = data.blanks;
                    svgContainer.innerHTML = svgFrame;
                    messages.innerHTML = '';
                    roundTag.innerText = `ROUND ${data.round}`;
                    guessInput.disabled = true;
                    guessInput.placeholder = "🎨 Drawer is planning sketch... Guessing locked!";
                    addMessage(`Round ${data.round} started! Target is ${data.length} letters.`, 'system');
                } else if (data.type === "round_active") {
                    guessInput.disabled = false;
                    guessInput.placeholder = "Type your guess here...";
                    guessInput.focus();
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
                    guessInput.disabled = true;
                    guessInput.placeholder = "You solved it! Waiting for round to finish...";
                } else if (data.type === "leaderboard") {
                    renderLeaderboard(data.roster);
                } else if (data.type === "round_end") {
                    wordBlanks.innerText = data.revealed_word.toUpperCase();
                    timerDisplay.innerText = "⏳ 0s";
                    progressBar.style.width = "0%";
                    guessInput.disabled = true;
                    addMessage(`Round over! The secret word was '${data.revealed_word}'.`, 'round-end-banner');
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
            if (guessInput.value.trim() && !guessInput.disabled) {
                ws.send(JSON.stringify({ type: "guess", text: guessInput.value.trim() }));
                guessInput.value = "";
            }
        }

        function startRandom() {
            const drawer = document.getElementById("drawer-select").value;
            fetch(`/start?random_pick=true&drawer=${encodeURIComponent(drawer)}`);
        }

        function startCustom() {
            const val = document.getElementById("custom-word").value.trim();
            if (val) {
                const drawer = document.getElementById("drawer-select").value;
                fetch(`/start?word=${encodeURIComponent(val)}&drawer=${encodeURIComponent(drawer)}`);
                document.getElementById("custom-word").value = "";
            }
        }
    </script>
</body>
</html>
"""

@app.get("/")
def get_ui():
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

            elif event_type == "guess":
                # Strict input gating: drop all guesses before strokes start
                if not game_state["is_running"] or not game_state["can_guess"]:
                    continue

                player = connected_players.get(websocket)
                if not player or player["solved"]:
                    continue

                guess = data.get("text", "").strip().lower()
                target = game_state["word"].lower()

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

                    # If all players (human + AI) have finished, complete the round immediately
                    all_humans_done = all(p["solved"] for p in connected_players.values())
                    if all_humans_done and game_state["ai_solved"]:
                        game_state["is_running"] = False
                elif is_close_guess(guess, target):
                    await websocket.send_json({
                        "type": "chat",
                        "sender": "close",
                        "text": f"'{guess}' is very close!"
                    })
                else:
                    # Echo guess publicly with the player's name
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

@app.get("/start")
async def start_game_round(word: str = None, random_pick: bool = False, drawer: str = DEFAULT_DRAWER):
    if game_state["is_running"]:
        return {"status": "A round is already running."}
    
    target_word = random.choice(WORD_BANK) if random_pick or not word else word.strip().lower()
    game_state["drawer"] = drawer
    asyncio.create_task(run_game_loop(target_word, drawer))
    return {"status": "started", "word_length": len(target_word)}

async def run_game_loop(secret_word: str, drawer_model: str):
    game_state["word"] = secret_word
    game_state["is_running"] = True
    game_state["can_guess"] = False
    game_state["round_num"] += 1
    game_state["ai_solved"] = False
    game_state["revealed_indices"] = set()
    game_state["time_left"] = 60

    for p in connected_players.values():
        p["solved"] = False

    attempted_ai_guesses = set()
    blanks = build_hint_pattern(secret_word, game_state["revealed_indices"])

    # 1. Lock input and broadcast preparation state
    await broadcast({
        "type": "prep_round",
        "round": game_state["round_num"],
        "length": len(secret_word),
        "blanks": blanks
    })
    await broadcast({"type": "leaderboard", "roster": get_leaderboard_payload()})
    await broadcast({"type": "chat", "sender": "system", "text": f"[{drawer_model}] is sketching..."})

    prompt = f"""You are playing Pictionary. Draw a highly detailed, progressive vector sketch of: '{secret_word}'.
Canvas: 300x300. Center is (150, 150).

DRAW IN 3 CHRONOLOGICAL PHASES (Output 12 to 16 total shapes):
- Phase 1 (Base & Foundation): Main silhouette, ground, or primary outline.
- Phase 2 (Surfaces & Colors): Main body volume, fills, branches, cushions, or rims.
- Phase 3 (Fine Details & Context): Spines/needles, wheels, buttons, texture lines, or background cues.

RULES FOR '{secret_word}':
1. Output 12 to 16 distinct geometric elements (<rect>, <circle>, <ellipse>, <path>, <line>).
2. Choose realistic colors matching '{secret_word}'. Use fill="..." for solid parts and stroke="..." for outlines.
3. Every shape must have stroke-width="3" or "4".
4. Return ONLY the raw <svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">...</svg> block. No text."""

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: ollama.chat(
            model=drawer_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
    )

    msg = response.get('message', {})
    svg_content = msg.get('content', '').strip()
    if not svg_content and 'thinking' in msg:
        svg_content = msg['thinking'].strip()

    strokes = sanitize_and_parse_strokes(svg_content)
    num_strokes = len(strokes)

    if not strokes:
        await broadcast({"type": "round_end", "revealed_word": secret_word})
        await broadcast({"type": "chat", "sender": "system", "text": "Drawer failed to generate recognizable shapes."})
        game_state["is_running"] = False
        return

    # Timeline calculation: 46s window for strokes
    DRAWING_WINDOW = 46.0
    stroke_schedule = [int(i * (DRAWING_WINDOW / num_strokes)) for i in range(num_strokes)]

    total_time = 60
    current_svg_body = ""
    stroke_idx = 0
    strokes_finished_announced = False

    # 2. Emit the first stroke and unlock guessing simultaneously
    current_svg_body += f"\n{strokes[0]}"
    stroke_idx = 1
    await broadcast({"type": "stroke_update", "svg": current_svg_body})
    game_state["can_guess"] = True
    await broadcast({"type": "round_active"})

    for second_left in range(total_time, 0, -1):
        if not game_state["is_running"]:
            break

        elapsed = total_time - second_left
        game_state["time_left"] = second_left
        await broadcast({"type": "timer_tick", "time_left": second_left})

        # Reveal 1st letter hint at 40s
        if second_left == 40 and len(secret_word) > 3 and 0 not in game_state["revealed_indices"]:
            game_state["revealed_indices"].add(0)
            await broadcast({"type": "hint_update", "blanks": build_hint_pattern(secret_word, game_state["revealed_indices"])})

        # Reveal 2nd letter hint at 20s
        if second_left == 20 and len(secret_word) > 4:
            avail = [i for i in range(1, len(secret_word)) if i not in game_state["revealed_indices"]]
            if avail:
                game_state["revealed_indices"].add(random.choice(avail))
                await broadcast({"type": "hint_update", "blanks": build_hint_pattern(secret_word, game_state["revealed_indices"])})

        # Timeline emission
        while stroke_idx < num_strokes and elapsed >= stroke_schedule[stroke_idx]:
            current_svg_body += f"\n{strokes[stroke_idx]}"
            await broadcast({"type": "stroke_update", "svg": current_svg_body})
            stroke_idx += 1
            if stroke_idx == num_strokes and not strokes_finished_announced:
                strokes_finished_announced = True
                await broadcast({"type": "chat", "sender": "system", "text": "🎨 Sketch complete! Keep guessing until time runs out!"})

        # Guesser attempts prediction every 4 seconds
        if not game_state["ai_solved"] and elapsed % 4 == 0 and current_svg_body:
            full_svg = f"""<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
                <style>
                    * {{ stroke-linecap: round; stroke-linejoin: round; }}
                    rect.canvas-bg {{ fill: white !important; stroke: none !important; }}
                </style>
                <rect class="canvas-bg" width="300" height="300"/>
                {current_svg_body}
            </svg>"""

            png_bytes = cairosvg.svg2png(bytestring=full_svg.encode("utf-8"), output_width=300, output_height=300)
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
                guess_prompt = f"What simple object is sketched in this image? The word has {len(secret_word)} letters. Answer with ONLY the single lowercase noun."

            ai_resp = await loop.run_in_executor(
                None, lambda: ollama.chat(
                    model=GUESSER_MODEL,
                    messages=[{"role": "user", "content": guess_prompt, "images": [png_bytes]}],
                    options={"temperature": 0.2}
                )
            )
            raw_guess = ai_resp['message']['content']
            ai_guess = re.sub(r"[^\w]", "", raw_guess).strip().lower()

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
                        game_state["is_running"] = False
                else:
                    if ai_guess not in attempted_ai_guesses:
                        attempted_ai_guesses.add(ai_guess)
                        await broadcast({"type": "chat", "sender": "ai", "text": f"[{GUESSER_MODEL}] guessed: {ai_guess}"})
            else:
                if ai_guess and ai_guess not in attempted_ai_guesses:
                    attempted_ai_guesses.add(ai_guess)
                    await broadcast({"type": "chat", "sender": "ai-warn", "text": f"[{GUESSER_MODEL}] tried: {ai_guess} (violates hint)"})

        await asyncio.sleep(1.0)

    # 3. Round wrap-up
    game_state["is_running"] = False
    game_state["can_guess"] = False
    await broadcast({"type": "round_end", "revealed_word": secret_word})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)