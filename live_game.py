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

COLOR_PALETTE = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#06b6d4"]

WORD_BANK = [
    "apple", "clock", "house", "car", "guitar", "bicycle", "airplane",
    "candle", "cactus", "chair", "umbrella", "bridge", "robot", "spider",
    "tree", "ladder", "camera", "boat", "cloud", "pizza", "sword", "snowman",
    "banana", "sun", "flower", "bottle", "glasses", "scissors"
]

active_connections: list[WebSocket] = []

# Persistent Game State across rounds
game_state = {
    "word": "",
    "drawer": DEFAULT_DRAWER,
    "is_running": False,
    "round_num": 0,
    "human_score": 0,
    "ai_score": 0,
    "human_guessed": False,
    "ai_guessed": False,
    "revealed_indices": set(),
    "time_left": 60
}

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates edit distance between two strings without external dependencies."""
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

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Pictionary - Live Arena</title>
    <link rel="icon" href="data:,">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; margin: 0; height: 100vh; background: #0b0f19; color: #f8fafc; }
        #canvas-panel { flex: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; border-right: 1px solid #1e293b; padding: 20px; }
        #chat-panel { flex: 1; display: flex; flex-direction: column; background: #070a10; }
        
        /* Scoreboard */
        #scoreboard { width: 360px; background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
        .score-box { display: flex; flex-direction: column; align-items: center; font-size: 13px; font-weight: 600; }
        .score-val { font-size: 17px; font-weight: 800; color: #38bdf8; }
        .status-badge { font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-top: 2px; }
        .badge-waiting { background: #374151; color: #9ca3af; }
        .badge-solved { background: #15803d; color: #86efac; font-weight: bold; }

        /* Timer and Blanks */
        #game-header { width: 360px; margin-bottom: 12px; display: flex; flex-direction: column; gap: 6px; }
        .status-row { display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 15px; }
        #timer-display { font-variant-numeric: tabular-nums; font-size: 18px; color: #38bdf8; }
        #word-blanks { font-family: monospace; font-size: 22px; letter-spacing: 5px; color: #fbbf24; font-weight: bold; }
        #progress-container { width: 100%; height: 6px; background: #1e293b; border-radius: 3px; overflow: hidden; }
        #progress-bar { width: 100%; height: 100%; background: #38bdf8; transition: width 1s linear, background-color 0.5s; }

        #canvas-container { background: #ffffff; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); overflow: hidden; width: 360px; height: 360px; display: flex; align-items: center; justify-content: center; }
        #messages { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 8px; font-size: 14px; }
        .msg { padding: 8px 12px; border-radius: 6px; max-width: 85%; }
        .human { background: #2563eb; align-self: flex-end; }
        .ai { background: #1e293b; border: 1px solid #334155; align-self: flex-start; }
        .ai-warn { background: #2d2218; border: 1px solid #78350f; color: #fbbf24; align-self: flex-start; font-size: 12px; }
        .system { color: #94a3b8; font-style: italic; align-self: center; font-size: 13px; }
        .hint { color: #fbbf24; font-weight: 600; align-self: center; font-size: 13px; }
        .close { background: #b45309; color: #fef3c7; font-weight: 600; align-self: flex-end; font-size: 13px; }
        .win-line { background: #15803d; color: #ffffff; font-weight: bold; align-self: center; text-align: center; width: 90%; }
        .round-end-banner { background: #1e1b4b; border: 1px solid #4338ca; color: #c7d2fe; font-weight: 600; align-self: center; text-align: center; width: 90%; }
        
        #input-box { display: flex; padding: 12px; border-top: 1px solid #1e293b; background: #070a10; }
        #guess-input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #fff; font-size: 14px; outline: none; }
        #guess-input:disabled { background: #1f2937; color: #6b7280; cursor: not-allowed; }
        
        #controls { margin-top: 15px; display: flex; flex-direction: column; gap: 10px; align-items: center; width: 360px; }
        .control-row { display: flex; gap: 8px; width: 100%; justify-content: center; align-items: center; }
        select, input[type="text"] { padding: 8px 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #fff; font-size: 13px; }
        button { padding: 8px 14px; border-radius: 6px; border: none; background: #2563eb; color: #fff; font-weight: 600; cursor: pointer; transition: background 0.15s; }
        button:hover { background: #1d4ed8; }
        button.action-btn { background: #059669; }
        button.action-btn:hover { background: #047857; }
    </style>
</head>
<body>
    <div id="canvas-panel">
        <div id="scoreboard">
            <div class="score-box">
                <span style="color: #94a3b8;">Round</span>
                <span class="score-val" id="round-counter">0</span>
            </div>
            <div class="score-box">
                <span>👤 Human</span>
                <span class="score-val" id="human-score">0</span>
                <span id="human-badge" class="status-badge badge-waiting">Waiting</span>
            </div>
            <div class="score-box">
                <span>🤖 AI</span>
                <span class="score-val" id="ai-score">0</span>
                <span id="ai-badge" class="status-badge badge-waiting">Waiting</span>
            </div>
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
            <svg id="live-svg" viewBox="0 0 300 300" width="360" height="360" xmlns="http://www.w3.org/2000/svg">
                <style>
                    * { fill: none !important; stroke-width: 4px !important; stroke-linecap: round; stroke-linejoin: round; }
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
            <input type="text" id="guess-input" placeholder="Type your guess here..." onkeydown="if(event.key==='Enter') sendGuess()"/>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        const svgContainer = document.getElementById("live-svg");
        const messages = document.getElementById("messages");
        const timerDisplay = document.getElementById("timer-display");
        const progressBar = document.getElementById("progress-bar");
        const wordBlanks = document.getElementById("word-blanks");
        const guessInput = document.getElementById("guess-input");

        const humanScoreEl = document.getElementById("human-score");
        const aiScoreEl = document.getElementById("ai-score");
        const humanBadge = document.getElementById("human-badge");
        const aiBadge = document.getElementById("ai-badge");
        const roundCounter = document.getElementById("round-counter");

        const svgFrame = `<style>
            * { fill: none !important; stroke-width: 4px !important; stroke-linecap: round; stroke-linejoin: round; }
            rect.canvas-bg { fill: #ffffff !important; stroke: none !important; }
        </style><rect class="canvas-bg" width="300" height="300"/>`;

        function addMessage(text, type) {
            const div = document.createElement("div");
            div.className = `msg ${type}`;
            div.innerText = text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "stroke_update") {
                svgContainer.innerHTML = svgFrame + data.svg;
            } else if (data.type === "chat") {
                addMessage(data.text, data.sender);
            } else if (data.type === "round_start") {
                wordBlanks.innerText = data.blanks;
                svgContainer.innerHTML = svgFrame;
                messages.innerHTML = '';
                roundCounter.innerText = data.round;
                humanBadge.className = "status-badge badge-waiting";
                humanBadge.innerText = "Waiting";
                aiBadge.className = "status-badge badge-waiting";
                aiBadge.innerText = "Waiting";
                guessInput.disabled = false;
                guessInput.placeholder = "Type your guess here...";
                guessInput.focus();
                addMessage(`--- Round ${data.round} Started! Guess the ${data.length}-letter object. ---`, 'system');
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
            } else if (data.type === "player_solved") {
                if (data.player === "human") {
                    humanBadge.className = "status-badge badge-solved";
                    humanBadge.innerText = "Solved ✓";
                    guessInput.disabled = true;
                    guessInput.placeholder = "You guessed the word! Waiting for round to finish...";
                } else if (data.player === "ai") {
                    aiBadge.className = "status-badge badge-solved";
                    aiBadge.innerText = "Solved ✓";
                }
            } else if (data.type === "score_update") {
                humanScoreEl.innerText = data.human_score;
                aiScoreEl.innerText = data.ai_score;
            } else if (data.type === "round_end") {
                wordBlanks.innerText = data.revealed_word.toUpperCase();
                timerDisplay.innerText = "⏳ 0s";
                progressBar.style.width = "0%";
                guessInput.disabled = true;
                addMessage(`Round finished! The secret word was '${data.revealed_word}'.`, 'round-end-banner');
            }
        };

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

async def broadcast(message: dict):
    for conn in active_connections:
        await conn.send_json(message)

def calculate_score(time_left: int, hints_revealed_count: int) -> int:
    """Timer-based scoring with penalty per revealed hint."""
    score = (time_left * 10) - (hints_revealed_count * 100)
    return max(50, score)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "guess" and game_state["is_running"]:
                guess = data["text"].strip().lower()
                target = game_state["word"].lower()

                if game_state["human_guessed"]:
                    continue

                if guess == target:
                    # Correct guess: compute points, do NOT expose secret word in chat
                    game_state["human_guessed"] = True
                    pts = calculate_score(game_state["time_left"], len(game_state["revealed_indices"]))
                    game_state["human_score"] += pts

                    await broadcast({
                        "type": "chat",
                        "sender": "win-line",
                        "text": f"🎉 You guessed the word! (+{pts} pts)"
                    })
                    await broadcast({"type": "player_solved", "player": "human"})
                    await broadcast({
                        "type": "score_update",
                        "human_score": game_state["human_score"],
                        "ai_score": game_state["ai_score"]
                    })

                    # If AI also already guessed, terminate round immediately
                    if game_state["ai_guessed"]:
                        game_state["is_running"] = False
                elif is_close_guess(guess, target):
                    # Near miss fuzzy warning
                    await broadcast({
                        "type": "chat",
                        "sender": "close",
                        "text": f"'{guess}' is very close!"
                    })
                else:
                    # Normal incorrect guess visible to all
                    await broadcast({"type": "chat", "sender": "human", "text": f"You guessed: {guess}"})

    except WebSocketDisconnect:
        active_connections.remove(websocket)

def parse_and_colorize_strokes(raw_svg: str) -> list[str]:
    svg_block = re.search(r"<svg[\s\S]*?<\/svg>", raw_svg, re.IGNORECASE)
    content = svg_block.group(0) if svg_block else raw_svg

    # Forgiving pattern matching both self-closing and unclosed tags
    stroke_pattern = r"(<(path|circle|rect|line|ellipse|polyline|polygon)\b[^>]*?(?:\/?>|>[\s\S]*?<\/\2>))"
    matches = [m[0] for m in re.findall(stroke_pattern, content, re.IGNORECASE)]
    
    clean = []
    for idx, s in enumerate(matches):
        if 'width="300"' in s and 'height="300"' in s:
            continue
        
        # Ensure tag is properly closed for cairosvg
        if not s.endswith("/>") and not re.search(r"<\/\w+>$", s):
            s = s.rstrip(">") + "/>"

        # Apply colorful stroke from palette if missing or monochrome
        chosen_color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        if "stroke=" not in s.lower():
            s = s.replace("/>", f' stroke="{chosen_color}"/>', 1)
        elif re.search(r'stroke=["\']?(black|#000|#111|#222|#333|gray)["\']?', s, re.IGNORECASE):
            s = re.sub(r'stroke=["\']?[^"\'>\s]+["\']?', f'stroke="{chosen_color}"', s)

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
    game_state["round_num"] += 1
    game_state["human_guessed"] = False
    game_state["ai_guessed"] = False
    game_state["revealed_indices"] = set()
    game_state["time_left"] = 60

    attempted_ai_guesses = set()

    blanks = build_hint_pattern(secret_word, game_state["revealed_indices"])
    await broadcast({
        "type": "round_start",
        "round": game_state["round_num"],
        "length": len(secret_word),
        "blanks": blanks
    })
    await broadcast({"type": "chat", "sender": "system", "text": f"[{drawer_model}] is sketching..."})

    # Restored few-shot prompt with explicit scale & orientation
    prompt = f"""You are playing Pictionary. Draw a clear, recognizable SVG outline sketch of: '{secret_word}'.
Canvas: 300x300. Center is (150, 150).

FEW-SHOT EXAMPLE:
<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
  <circle cx="150" cy="140" r="50" stroke="#10b981" stroke-width="4" fill="none"/>
  <line x1="150" y1="190" x2="150" y2="260" stroke="#8b5cf6" stroke-width="6"/>
</svg>

SPATIAL ORIENTATION RULES:
- For HORIZONTAL or VEHICLE objects (bicycle, car, airplane, boat, glasses):
  Spread components left-to-right along the X-axis (e.g. wheels at cx="80" and cx="220").
- For VERTICAL objects (tree, candle, snowman, sword, ladder):
  Stack components along the Y-axis.

RULES FOR '{secret_word}':
1. Output 4 to 8 distinct geometric shapes (circle, line, rect, path).
2. Canvas scale: main structure 80-140px centered around (150, 150).
3. Use stroke colors (e.g. stroke="#3b82f6", stroke="#ef4444", stroke="#10b981").
4. Every shape MUST have fill="none" and stroke-width="4".
5. Return ONLY the raw <svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">...</svg> block."""

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: ollama.chat(
            model=drawer_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.25}
        )
    )

    msg = response.get('message', {})
    svg_content = msg.get('content', '').strip()
    
    # Fallback if reasoning tokens were placed in thinking buffer
    if not svg_content and 'thinking' in msg:
        svg_content = msg['thinking'].strip()

    print(f"\n==========================================")
    print(f"[DEBUG Drawer Raw ({drawer_model}) for '{secret_word}']:")
    print(svg_content if svg_content else f"<EMPTY> | Done: {response.get('done_reason')} | Count: {response.get('eval_count')}")
    print(f"==========================================")

    strokes = parse_and_colorize_strokes(svg_content)
    print(f"[DEBUG Drawer Parsed]: Extracted {len(strokes)} valid strokes from {drawer_model}\n")

    if not strokes:
        await broadcast({"type": "round_end", "revealed_word": secret_word})
        await broadcast({"type": "chat", "sender": "system", "text": "Drawer failed to generate recognizable shapes."})
        game_state["is_running"] = False
        return

    # Master 60-Second Loop
    total_time = 60
    current_svg_body = ""
    stroke_idx = 0
    num_strokes = len(strokes)
    strokes_finished_announced = False

    for second_left in range(total_time, 0, -1):
        if not game_state["is_running"]:
            break

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

        # Progressive stroke reveals
        if stroke_idx < num_strokes and (total_time - second_left) % 3 == 0:
            current_svg_body += f"\n{strokes[stroke_idx]}"
            await broadcast({"type": "stroke_update", "svg": current_svg_body})
            stroke_idx += 1
            if stroke_idx == num_strokes and not strokes_finished_announced:
                strokes_finished_announced = True
                await broadcast({"type": "chat", "sender": "system", "text": "🎨 Sketch complete! Keep guessing until time runs out!"})

        # Guesser attempts prediction every 4 seconds (if AI has not already guessed)
        if not game_state["ai_guessed"] and (total_time - second_left) % 4 == 0 and current_svg_body:
            full_svg = f"""<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
                <style>
                    * {{ fill: none !important; stroke-width: 4px !important; stroke-linecap: round; stroke-linejoin: round; }}
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
            print(f"[DEBUG Guesser Raw] '{ai_guess}' for target '{secret_word}' with pattern '{current_pattern}'")

            if matches_pattern(ai_guess, secret_word, game_state["revealed_indices"]):
                if ai_guess == secret_word.lower():
                    # AI guessed correctly: award points and conceal word
                    game_state["ai_guessed"] = True
                    pts = calculate_score(second_left, len(game_state["revealed_indices"]))
                    game_state["ai_score"] += pts

                    await broadcast({
                        "type": "chat",
                        "sender": "win-line",
                        "text": f"🤖 [{GUESSER_MODEL}] guessed the word! (+{pts} pts)"
                    })
                    await broadcast({"type": "player_solved", "player": "ai"})
                    await broadcast({
                        "type": "score_update",
                        "human_score": game_state["human_score"],
                        "ai_score": game_state["ai_score"]
                    })

                    # If Human also solved, round is complete
                    if game_state["human_guessed"]:
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

    # Round wrap-up
    game_state["is_running"] = False
    await broadcast({"type": "round_end", "revealed_word": secret_word})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)