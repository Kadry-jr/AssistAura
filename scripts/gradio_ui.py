import requests
import gradio as gr
import uuid

# ========================
# CONFIG
# ========================
BACKEND_URL = "http://127.0.0.1:8000/api/chat"


# ========================
# HELPER
# ========================
def chat_with_backend(message, session_id, k=5):
    payload = {"session_id": session_id, "query": message, "k": k}
    try:
        response = requests.post(BACKEND_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "No answer from backend.")
        hits = data.get("hits", [])
        session_id = data.get("session_id", session_id)
        return answer, session_id, hits
    except Exception as e:
        return f"⚠️ Error contacting backend: {str(e)}", session_id, []


# ========================
# CHAT LOGIC (patched)
# ========================
def user_message(user_message, history, session_id, k):
    bot_message, session_id, hits = chat_with_backend(user_message, session_id, k)

    if history is None:
        history = []

    # Detect format (tuple or dict)
    uses_tuple_style = False
    if len(history) > 0 and isinstance(history[0], (list, tuple)):
        uses_tuple_style = True

    if uses_tuple_style:
        history.append((user_message, bot_message))
    else:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_message})

    # Build property cards
    cards = []
    for hit in hits:
        meta = hit.get("metadata", {}) or {}

        title = meta.get("title") or "🏠 Property"
        project = meta.get("project")
        location = meta.get("location") or meta.get("city") or "Unknown"
        beds = meta.get("beds")
        baths = meta.get("baths")
        area = meta.get("area_m2") or meta.get("area") or None

        price_raw = (
            meta.get("price_egp")
            or meta.get("price")
            or meta.get("price_formatted")
        )
        if price_raw:
            if isinstance(price_raw, (int, float)):
                price_display = f"{price_raw:,.0f} EGP"
            else:
                price_display = str(price_raw)
        else:
            price_display = "N/A"

        url = meta.get("url")

        lines = [f"### {title}"]
        if project:
            lines.append(f"🏗️ **Project:** {project}")
        lines.append(f"📍 **Location:** {location}")
        if beds or baths:
            lines.append(f"🛏️ {beds or '?'} beds | 🛁 {baths or '?'} baths")
        if area:
            lines.append(f"📐 **Area:** {area} sqm")
        lines.append(f"💰 **Price:** {price_display}")
        if url:
            lines.append(f"🔗 [View Listing]({url})")

        cards.append("\n\n".join(lines))

    hits_markdown = "\n\n---\n\n".join(cards) if cards else "No relevant properties found."

    return history, "", session_id, hits_markdown


# ========================
# BUILD UI
# ========================
custom_css = """
body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #1e293b, #0f766e);
    color: #f8fafc;
}
.gradio-container {
    max-width: 1000px !important;
    margin: auto;
}
.chatbox {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    border-radius: 14px;
    padding: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    font-size: 1.1rem;
}
.resultsbox {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    max-height: 500px;  /* ≈ fits 5 cards */
    overflow-y: auto;
    font-size: 1.05rem;
}
.resultsbox::-webkit-scrollbar {
    width: 10px;
}
.resultsbox::-webkit-scrollbar-thumb {
    background: #22c55e;
    border-radius: 10px;
}
.resultsbox::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.1);
}
.message.user {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 14px;
    padding: 10px 14px;
    font-size: 1.05rem;
}
.message.assistant {
    background-color: #16a34a !important;
    color: white !important;
    border-radius: 14px;
    padding: 10px 14px;
    font-size: 1.05rem;
}
"""

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="teal", neutral_hue="slate"),
    title="🏡 AssistAura — Real Estate Chatbot"
) as demo:

    gr.HTML(
        """
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
        <div style="text-align:center; padding:1.5rem; border-radius:16px;
             background:linear-gradient(90deg,#22c55e,#14b8a6); color:white; margin-bottom:1.5rem;">
            <h1 style="margin:0; font-size:2.5rem;">🏡 AssistAura</h1>
            <p style="font-size:1.2rem;">Your AI-powered real estate assistant in <b>Egypt</b>.  
            Discover properties, prices, and neighborhoods in style!</p>
        </div>
        """
    )

    with gr.Row():
        # Chat area
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat", elem_classes="section-title")
            chatbot = gr.Chatbot(
                type="messages", height=600,
                bubble_full_width=False, elem_classes="chatbox"
            )
            msg = gr.Textbox(
                placeholder="💡 Ask me about apartments, prices, or locations...",
                show_label=False, autofocus=True
            )
            with gr.Row():
                clear = gr.Button("🧹 Clear Chat", variant="secondary")
                k_slider = gr.Slider(
                    1, 10, value=5, step=1,
                    label="📊 Number of property results",
                    interactive=True
                )

        # Property results
        with gr.Column(scale=1):
            gr.Markdown("### 🔎 Relevant Properties", elem_classes="section-title")
            hits_display = gr.Markdown(
                "✨ No results yet. Start asking about properties!",
                elem_classes="resultsbox"
            )

    session_state = gr.State(str(uuid.uuid4()))

    msg.submit(
        user_message,
        [msg, chatbot, session_state, k_slider],
        [chatbot, msg, session_state, hits_display]
    )

    clear.click(
        lambda: ([], str(uuid.uuid4()), "✨ No results yet. Start asking about properties!"),
        None,
        [chatbot, session_state, hits_display],
        queue=False
    )

demo.launch(server_name="127.0.0.1", server_port=7860)
