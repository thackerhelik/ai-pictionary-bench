import asyncio
import io
import re
import time
import cairosvg
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import HTMLResponse

app = FastAPI()

# Available models in your local Ollama setup
DEFAULT_DRAWER = "qwen3.5:2b"  # 3.5 generally has better code grounding than 2.5
GUESSER_MODEL = "qwen2.5vl:3b"

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
    <title>AI Pictionary - Live Arena</title>
    <link rel="icon" href="data:,">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; margin: 0; height: 100vh; background: #0f172a; color: #f8fafc; }
        #canvas-panel { flex: 2; display: flex; flex-direction: column; align-items: center; justify-content: center; border-right: 1px solid #1e293b; padding: 20px; }
        #chat-panel { flex: 1; display: flex; flex-direction: column; background: #0b0f19; }
        #canvas-container { background: #ffffff; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); overflow: hidden; width: 350px; height: 350px; display: flex; align-items: center; justify-content: center; }
        #messages { flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 8px; font-size: 14px; }
        .msg { padding: 8px 12px; border-radius: 6px; max-width: 85%; }
        .human { background: #2563eb; align-self: flex-end; }
        .ai { background: #334155; align-self: flex-start; }
        .system { color: #94a3b8; font-style: italic; align-self: center; font-size: 13px; }
        .win { background: #16a34a; font-weight: bold; align-self: center; }
        #input-box { display: flex; padding: 12px; border-top: 1px solid #1e293b; background: #0b0f19; }
        #guess-input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #1e293b; color: #fff; font-size: 14px; outline: none; }
        #controls { margin-top: 15px; display: flex; flex-direction: column; gap: 10px; align-items: center; }
        .btn-group { display: flex; gap: 8px; }
        .custom-group { display: flex; gap: 6px; align-items: center; }
        select, input[type="text"] { padding: 7px 10px; border-radius: 6px; border: 1px solid #334155; background: #1e293b; color: #fff; font-size: 13px; }
        button { padding: 7px 14px; border-radius: 6px; border: none; background: #2563eb; color: #fff; font-weight: 600; cursor: pointer; }
        button:hover { background: #1d4ed8; }
    </style>
</head>
<body>
    <div id="canvas-panel">
        <h2 id="round-info" style="margin-bottom: 12px;">Pick a word to start drawing</h2>
        <div id="canvas-container">
            <svg id="live-svg" viewBox="0 0 300 300" width="350" height="350" xmlns="http://www.w3.org/2000/svg">
                <style>
                    * { fill: none !important; stroke: #111111 !important; stroke-width: 4px !important; stroke-linecap: round; stroke-linejoin: round; }
                    rect.canvas-bg { fill: #ffffff !important; stroke: none !important; }
                </style>
                <rect class="canvas-bg" width="300" height="300"/>
            </svg>
        </div>
        <div id="controls">
            <div class="custom-group">
                <label style="font-size: 13px; color: #94a3b8;">Drawer Model:</label>
                <select id="drawer-select">
                    <option value="qwen3.5:2b">qwen3.5:2b</option>
                    <option value="qwen2.5:3b">qwen2.5:3b</option>
                    <option value="gemma4:e4b">gemma4:e4b</option>
                </select>
            </div>
            <div class="btn-group">
                <button onclick="startRound('apple')">Apple</button>
                <button onclick="startRound('clock')">Clock</button>
                <button onclick="startRound('house')">House</button>
                <button onclick="startRound('car')">Car</button>
            </div>
            <div class="custom-group">
                <input type="text" id="custom-word" placeholder="Custom word..." />
                <button onclick="startCustom()">Draw Custom</button>
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

        const svgFrame = `<style>
            * { fill: none !important; stroke: #111111 !important; stroke-width: 4px !important; stroke-linecap: round; stroke-linejoin: round; }
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
                document.getElementById("round-info").innerText = `Target: ${data.length} letters`;
                svgContainer.innerHTML = svgFrame;
                messages.innerHTML = '';
                addMessage(`New Round! Guess the ${data.length}-letter object.`, 'system');
            }
        };

        function sendGuess() {
            const input = document.getElementById("guess-input");
            if (input.value.trim()) {
                ws.send(JSON.stringify({ type: "guess", text: input.value.trim() }));
                input.value = "";
            }
        }

        function startRound(word) {
            const drawer = document.getElementById("drawer-select").value;
            fetch(`/start?word=${encodeURIComponent(word)}&drawer=${encodeURIComponent(drawer)}`);
        }

        function startCustom() {
            const val = document.getElementById("custom-word").value.trim();
            if (val) {
                startRound(val);
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
                    await broadcast({"type": "chat", "sender": "win", "text": f"🎉 YOU WON! The word was '{game_state['word']}'!"})
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast(message: dict):
    for conn in active_connections:
        await conn.send_json(message)

def parse_svg_strokes(raw_svg: str) -> list[str]:
    # Extract distinct geometric tags
    stroke_pattern = r"(<(path|circle|rect|line|ellipse|polyline|polygon)[^>]*?(?:\/>|>[\s\S]*?<\/\2>))"
    matches = [m[0] for m in re.findall(stroke_pattern, raw_svg, re.IGNORECASE)]
    
    clean_strokes = []
    for s in matches:
        # Ignore canvas backgrounds
        if 'width="300"' in s and 'height="300"' in s:
            continue
        # Ignore tiny dot circles (radius < 12px)
        r_match = re.search(r'\br=["\']?(\d+)', s)
        if r_match and int(r_match.group(1)) < 12:
            continue
        clean_strokes.append(s)
        
    return clean_strokes

@app.get("/start")
async def start_game_round(word: str, drawer: str = DEFAULT_DRAWER):
    if game_state["is_running"]:
        return {"status": "A round is already running."}
    
    game_state["drawer"] = drawer
    asyncio.create_task(run_game_loop(word.strip(), drawer))
    return {"status": "started"}

async def run_game_loop(secret_word: str, drawer_model: str):
    game_state["word"] = secret_word
    game_state["is_running"] = True

    await broadcast({"type": "round_start", "length": len(secret_word)})
    await broadcast({"type": "chat", "sender": "system", "text": f"[{drawer_model}] is sketching '{secret_word}'..."})

    # Few-shot prompt with explicit scale anchoring
    prompt = f"""You are playing Pictionary. Draw a clear, recognizable outline sketch of: '{secret_word}'.
Canvas: 300x300. Center is (150, 150).

FEW-SHOT EXAMPLE for 'mug':
<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
  <rect x="100" y="110" width="90" height="110" rx="15" fill="none" stroke="black" stroke-width="4"/>
  <path d="M 190 130 C 230 130 230 190 190 190" fill="none" stroke="black" stroke-width="4"/>
  <ellipse cx="145" cy="110" rx="45" ry="12" fill="none" stroke="black" stroke-width="4"/>
</svg>

RULES FOR '{secret_word}':
1. Output 3 to 6 distinct, visible parts (circle, rect, path, line).
2. Use large coordinates: main body width/height or radius must be 50 to 90 pixels (NEVER draw tiny dots).
3. Every element must have fill="none" stroke="black" stroke-width="4".
4. Return ONLY the raw <svg>...</svg> block. No text explanations."""

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, lambda: ollama.chat(
            model=drawer_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3}
        )
    )

    svg_content = response['message']['content']
    print(f"\n--- [DEBUG] Raw Drawer Output ({drawer_model}) ---")
    print(svg_content)
    
    strokes = parse_svg_strokes(svg_content)
    print(f"--- [DEBUG] Parsed Strokes ({len(strokes)} found) ---")
    for i, s in enumerate(strokes):
        print(f"Stroke {i+1}: {s}")

    if not strokes:
        await broadcast({"type": "chat", "sender": "system", "text": "Drawer failed to generate recognizable strokes. Check terminal."})
        game_state["is_running"] = False
        return

    current_svg_body = ""
    for i, stroke in enumerate(strokes):
        if not game_state["is_running"]:
            break

        current_svg_body += f"\n{stroke}"
        await broadcast({"type": "stroke_update", "svg": current_svg_body})
        await asyncio.sleep(2.0)

        # Cairo rendering wrapper
        full_svg = f"""<svg viewBox="0 0 300 300" xmlns="http://www.w3.org/2000/svg">
            <style>
                * {{ fill: none !important; stroke: black !important; stroke-width: 4px !important; stroke-linecap: round; stroke-linejoin: round; }}
                rect.canvas-bg {{ fill: white !important; stroke: none !important; }}
            </style>
            <rect class="canvas-bg" width="300" height="300"/>
            {current_svg_body}
        </svg>"""

        png_bytes = cairosvg.svg2png(bytestring=full_svg.encode("utf-8"), output_width=300, output_height=300)

        guess_prompt = f"What object is drawn here? The word has {len(secret_word)} letters. Answer with ONLY the single lowercase noun. Never output letters or digits."
        ai_resp = await loop.run_in_executor(
            None, lambda: ollama.chat(
                model=GUESSER_MODEL,
                messages=[{"role": "user", "content": guess_prompt, "images": [png_bytes]}],
                options={"temperature": 0.1}
            )
        )
        ai_guess = re.sub(r"[^\w]", "", ai_resp['message']['content']).strip().lower()
        await broadcast({"type": "chat", "sender": "ai", "text": f"[{GUESSER_MODEL}] guessed: {ai_guess}"})

        if ai_guess == secret_word.lower():
            game_state["is_running"] = False
            await broadcast({"type": "chat", "sender": "win", "text": f"🤖 AI WON! Identified '{secret_word}' on stroke {i+1}!"})
            break

    if game_state["is_running"]:
        game_state["is_running"] = False
        await broadcast({"type": "chat", "sender": "system", "text": f"Time's up! The word was '{secret_word}'."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)