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

# Game dictionary
WORD_BANK = [
    "apple", "clock", "house", "car", "guitar", "bicycle", "airplane",
    "candle", "cactus", "chair", "umbrella", "bridge", "robot", "spider",
    "tree", "ladder", "camera", "boat", "cloud", "pizza", "sword", "snowman",
    "banana", "sun", "flower", "bottle", "glasses", "scissors"
]

active_connections: list[WebSocket] = []
game_state = {
    "word": "",
    "drawer": DEFAULT_DRAWER,
    "is_running": False
}

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI Pictionary - Color Arena</title>
    <link rel="icon" href="data:,">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; margin: 0; height: 100vh; background: #0b0f19; color: #f8fafc; }
        #canvas-panel { flex: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; border-right: 1px solid #1e293b; padding: 20px; }
        #chat-panel { flex: 1; display: flex; flex-direction: column; background: #070a10; }
        
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
        .win { background: #16a34a; font-weight: bold; align-self: center; text-align: center; }
        
        #input-box { display: flex; padding: 12px; border-top: 1px solid #1e293b; background: #070a10; }
        #guess-input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #111827; color: #fff; font-size: 14px; outline: none; }
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
                addMessage(`New game! Target word has ${data.length} letters.`, 'system');
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
            } else if (data.type === "round_end") {
                wordBlanks.innerText = data.revealed_word.toUpperCase();
                timerDisplay.innerText = "⏳ 0s";
                progressBar.style.width = "0%";
            }
        };

        function sendGuess() {
            const input = document.getElementById("guess-input");
            if (input.value.trim()) {
                ws.send(JSON.stringify({ type: "guess", text: input.value.trim() }));
                input.value = "";
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
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "guess" and game_state["is_running"]:
                guess = data["text"].strip().lower()
                await broadcast({"type": "chat", "sender": "human", "text": f"You guessed: {guess}"})
                if guess == game_state["word"].lower():
                    game_state["is_running"] = False
                    await broadcast({"type": "round_end", "revealed_word": game_state["word"]})
                    await broadcast({"type": "chat", "sender": "win", "text": f"🎉 YOU WON! You correctly identified '{game_state['word']}'!"})
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast(message: dict):
    for conn in active_connections:
        await conn.send_json(message)

def parse_and_colorize_strokes(raw_svg: str) -> list[str]:
    stroke_pattern = r"(<(path|circle|rect|line|ellipse|polyline|polygon)[^>]*?(?:\/>|>[\s\S]*?<\/\2>))"
    matches = [m[0] for m in re.findall(stroke_pattern, raw_svg, re.IGNORECASE)]
    
    clean = []
    for idx, s in enumerate(matches):
        if 'width="300"' in s and 'height="300"' in s:
            continue
        chosen_color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        if "stroke=" not in s.lower():
            s = s.replace(">", f' stroke="{chosen_color}">', 1)
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
    revealed_indices = set()
    attempted_ai_guesses = set()

    blanks = build_hint_pattern(secret_word, revealed_indices)
    await broadcast({"type": "round_start", "length": len(secret_word), "blanks": blanks})
    await broadcast({"type": "chat", "sender": "system", "text": f"[{drawer_model}] is sketching..."})

    # Drawer prompt with explicit horizontal vs vertical spatial layout rules
    prompt = f"""You are playing Pictionary. Draw a clear, recognizable SVG outline sketch of: '{secret_word}'.
Canvas: 300x300. Center is (150, 150).

SPATIAL ORIENTATION RULES:
- For HORIZONTAL or VEHICLE objects (bicycle, car, airplane, boat, glasses):
  Spread components left-to-right along the X-axis!
  For 'bicycle': draw left wheel at cx="80" cy="200" r="45", right wheel at cx="220" cy="200" r="45", and connect them with frame lines between (80, 200) and (220, 200). DO NOT stack wheels on top of each other!
- For VERTICAL objects (tree, candle, snowman, sword):
  Stack components along the Y-axis.

GENERAL RULES:
1. Output 5 to 9 distinct geometric shapes (circle, path, rect, line).
2. Use bright stroke colors (e.g. stroke="#3b82f6", stroke="#ef4444", stroke="#10b981").
3. Every shape MUST have fill="none" and stroke-width="4".
4. Return ONLY raw SVG inside <svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">...</svg>. No explanations."""

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: ollama.chat(
            model=drawer_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2}
        )
    )

    svg_content = response['message']['content']
    strokes = parse_and_colorize_strokes(svg_content)

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

        await broadcast({"type": "timer_tick", "time_left": second_left})

        # Reveal 1st letter hint at 40s
        if second_left == 40 and len(secret_word) > 3 and 0 not in revealed_indices:
            revealed_indices.add(0)
            await broadcast({"type": "hint_update", "blanks": build_hint_pattern(secret_word, revealed_indices)})

        # Reveal 2nd letter hint at 20s
        if second_left == 20 and len(secret_word) > 4:
            avail = [i for i in range(1, len(secret_word)) if i not in revealed_indices]
            if avail:
                revealed_indices.add(random.choice(avail))
                await broadcast({"type": "hint_update", "blanks": build_hint_pattern(secret_word, revealed_indices)})

        # Progressive stroke drawing
        if stroke_idx < num_strokes and (total_time - second_left) % 3 == 0:
            current_svg_body += f"\n{strokes[stroke_idx]}"
            await broadcast({"type": "stroke_update", "svg": current_svg_body})
            stroke_idx += 1
            if stroke_idx == num_strokes and not strokes_finished_announced:
                strokes_finished_announced = True
                await broadcast({"type": "chat", "sender": "system", "text": "🎨 Sketch complete! Keep guessing until time runs out!"})

        # Guesser attempts prediction every 4 seconds
        if (total_time - second_left) % 4 == 0 and current_svg_body:
            full_svg = f"""<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
                <style>
                    * {{ fill: none !important; stroke-width: 4px !important; stroke-linecap: round; stroke-linejoin: round; }}
                    rect.canvas-bg {{ fill: white !important; stroke: none !important; }}
                </style>
                <rect class="canvas-bg" width="300" height="300"/>
                {current_svg_body}
            </svg>"""

            png_bytes = cairosvg.svg2png(bytestring=full_svg.encode("utf-8"), output_width=300, output_height=300)
            current_pattern = build_hint_pattern(secret_word, revealed_indices)

            # Dictionary-assisted candidate list when hints exist
            if revealed_indices:
                candidates = get_candidate_words(secret_word, revealed_indices)
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

            # Validate against pattern
            if matches_pattern(ai_guess, secret_word, revealed_indices):
                if ai_guess not in attempted_ai_guesses:
                    attempted_ai_guesses.add(ai_guess)
                    await broadcast({"type": "chat", "sender": "ai", "text": f"[{GUESSER_MODEL}] guessed: {ai_guess}"})
                    if ai_guess == secret_word.lower():
                        game_state["is_running"] = False
                        await broadcast({"type": "round_end", "revealed_word": secret_word})
                        await broadcast({"type": "chat", "sender": "win", "text": f"🤖 AI WON! Correctly identified '{secret_word}'!"})
                        break
            else:
                # If it guessed a word violating revealed hints, show it with a warning rather than going silent
                if ai_guess and ai_guess not in attempted_ai_guesses:
                    attempted_ai_guesses.add(ai_guess)
                    await broadcast({"type": "chat", "sender": "ai-warn", "text": f"[{GUESSER_MODEL}] tried: {ai_guess} (violates hint)"})

        await asyncio.sleep(1.0)

    if game_state["is_running"]:
        game_state["is_running"] = False
        await broadcast({"type": "round_end", "revealed_word": secret_word})
        await broadcast({"type": "chat", "sender": "system", "text": f"Time's up! The secret word was '{secret_word}'."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)