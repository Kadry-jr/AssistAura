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

        # Create property card with better formatting
        card_html = f"""
        <div class="property-card">
            <div class="property-header">
                <h3>{title}</h3>
            </div>
            <div class="property-details">
                {f'<p class="project"><span class="icon">🏗️</span> <strong>Project:</strong> {project}</p>' if project else ''}
                <p class="location"><span class="icon">📍</span> <strong>Location:</strong> {location}</p>
                {f'<p class="rooms"><span class="icon">🛏️</span> {beds or "?"} beds | <span class="icon">🛁</span> {baths or "?"} baths</p>' if beds or baths else ''}
                {f'<p class="area"><span class="icon">📐</span> <strong>Area:</strong> {area} sqm</p>' if area else ''}
                <p class="price"><span class="icon">💰</span> <strong>Price:</strong> {price_display}</p>
                {f'<p class="link"><a href="{url}" target="_blank" class="view-link">🔗 View Listing</a></p>' if url else ''}
            </div>
        </div>
        """
        cards.append(card_html)

    hits_markdown = "\n\n".join(cards) if cards else '<div class="no-results">✨ No relevant properties found.</div>'

    return history, "", session_id, hits_markdown


# ========================
# BUILD UI
# ========================
custom_css = """/* Import Google Fonts - must be first */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Make all text white across the app */
* {
  color: white !important;
}

/* Make input placeholder text white */
input::placeholder,
textarea::placeholder {
  color: white !important;
  opacity: 1;
}

/* Root Variables */
:root {
  --primary-color: #14283C;
  --secondary-color: #B4A788;
  --accent-color: #FFFFFF;
  --background-dark: #14283C;
  --background-light: #B4A788;
  --text-light: #FFFFFF;
  --card-bg: rgba(255,255,255,0.06);
  --border-radius: 16px;
  --shadow: 0 8px 32px rgba(0,0,0,0.32);
}

/* Basic reset */
* { box-sizing: border-box; }
html, body { height: 100%; }

/* Body + Container */
body, .gradio-container {
  font-family: 'Poppins', sans-serif !important;
  background: linear-gradient(135deg, var(--background-dark), var(--background-light)) !important;
  color: var(--text-light) !important;
  margin: 0;
  padding: 0;
}

.gradio-container {
  max-width: 1400px !important;
  margin: 0 auto !important;
  padding: 1rem !important;
}

/* Header */
.header {
  text-align: center;
  padding: 0.8rem 1rem;
  background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
  margin: 0 auto 1rem auto;
  border-radius: 12px;
  width: 100%;
  max-width: 1100px;
}
.header h1 {
  margin: 0;
  font-size: 1.9rem;
  font-weight: 700;
  color: white;
}
.header p {
  margin: 0.25rem 0 0 0;
  font-size: 1rem;
  color: rgba(255,255,255,0.95);
}

/* ===== Chat + Results containers ===== */
.chatbot-container, .chatbox {
  background: var(--card-bg) !important;
  backdrop-filter: blur(12px) !important;
  border-radius: var(--border-radius) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  box-shadow: var(--shadow) !important;
  display: flex;
  flex-direction: column;
  gap: 0.6rem;
  padding: 1rem;
  height: 700px !important; /* bigger chat */
  overflow-y: auto;
}

.results-container, .resultsbox {
  background: var(--card-bg) !important;
  backdrop-filter: blur(12px) !important;
  border-radius: var(--border-radius) !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  padding: 1rem !important;
  max-height: 600px !important; /* slightly smaller */
  overflow-y: auto !important;
  box-shadow: var(--shadow) !important;
}

/* Scrollbars */
.results-container::-webkit-scrollbar,
.chatbot-container::-webkit-scrollbar,
.resultsbox::-webkit-scrollbar,
.chatbox::-webkit-scrollbar {
  width: 8px;
}
.results-container::-webkit-scrollbar-thumb,
.chatbot-container::-webkit-scrollbar-thumb,
.resultsbox::-webkit-scrollbar-thumb,
.chatbox::-webkit-scrollbar-thumb {
  background: linear-gradient(45deg, var(--primary-color), var(--secondary-color));
  border-radius: 8px;
}
.results-container::-webkit-scrollbar-track,
.chatbot-container::-webkit-scrollbar-track {
  background: rgba(255,255,255,0.03);
}

/* ===== Messages ===== */
.message {
  display: block;
  margin: 0.6rem 0;
  padding: 12px 16px;
  border-radius: 14px;
  line-height: 1.5;
  font-size: 1rem;
  word-wrap: break-word;
  white-space: pre-wrap;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  max-width: 95% !important; /* wider messages */
  min-width: 200px !important;
}

/* User message */
.message.user {
  align-self: flex-end;
  background: linear-gradient(180deg, #2563eb, #1e40af) !important;
  color: #fff !important;
  border-bottom-right-radius: 6px !important;
  text-align: left !important;
}

/* Assistant message */
.message.assistant {
  align-self: flex-start;
  background: linear-gradient(180deg, #16a34a, #0f6f3f) !important;
  color: #fff !important;
  border-bottom-left-radius: 6px !important;
  text-align: left !important;
}

/* Disable animations */
.chatbox .message * {
  animation: none !important;
  transition: none !important;
}

/* ===== Input styling ===== */
.chat-input-row, .input-row {
  display: flex !important;
  gap: 0.75rem !important;
  align-items: center !important;
  width: 100% !important;
  padding: 0 0.25rem !important;
  margin-top: 0.5rem !important;
}

.input-field, .textbox, textarea, input[type="text"] {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
  color: var(--text-light) !important;
  padding: 14px 16px !important;
  font-size: 1.05rem !important;
  min-height: 52px !important;
  width: 100% !important;
}
.input-field::placeholder,
.textbox::placeholder {
  color: rgba(255,255,255,0.7) !important;
}
.input-field:focus,
.textbox:focus {
  border-color: var(--secondary-color) !important;
  box-shadow: 0 0 0 3px rgba(180,167,136,0.25) !important;
  outline: none !important;
}

/* Buttons */
.button-primary, .gr-button, button {
  background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 8px 16px !important;
  font-weight: 600 !important;
  cursor: pointer !important;
}
.button-primary:hover, .gr-button:hover, button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.24);
}

/* ===== Property cards ===== */
.property-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px;
  padding: 0.8rem !important;   /* compact */
  margin-bottom: 0.75rem !important;
  font-size: 0.9rem !important;
  transition: transform 0.22s ease, box-shadow 0.22s ease;
  backdrop-filter: blur(6px);
}
.property-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.property-header h3 {
  margin: 0 0 0.6rem 0;
  color: var(--accent-color);
  font-size: 1rem !important;
  font-weight: 600;
}
.property-details p {
  margin: 0.4rem 0;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.85rem !important;
  color: rgba(255,255,255,0.92);
}
.property-details .icon {
  width: 18px;
  display: inline-block;
  text-align: center;
}
.property-details .price {
  font-weight: 700;
  font-size: 1rem;
  color: var(--secondary-color);
}

/* View link */
.view-link {
  color: var(--primary-color) !important;
  text-decoration: none !important;
  font-weight: 600 !important;
  padding: 0.4rem 0.8rem !important;
  border-radius: 8px !important;
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.03) !important;
}

/* No results */
.no-results {
  text-align: center;
  padding: 2rem 1rem;
  color: rgba(255,255,255,0.7);
  font-size: 1.05rem;
  font-style: italic;
  border-radius: 10px;
}

/* Section titles */
.section-title {
  color: var(--accent-color) !important;
  font-weight: 600 !important;
  font-size: 1.15rem !important;
  margin-bottom: 0.8rem !important;
  display: flex !important;
  align-items: center !important;
  gap: 0.5rem !important;
}

/* Responsive */
@media (max-width: 900px) {
  .gradio-container { padding: 0.5rem !important; }
  .chatbot-container, .chatbox, .results-container, .resultsbox {
    height: auto !important;
    max-height: 60vh !important;
  }
  .header h1 { font-size: 1.4rem; }
  .header p { display: none; }
  .message { max-width: 100% !important; min-width: 0 !important; }
}

/* Simple fade animation */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.property-card { animation: fadeIn 0.35s ease-out; }


"""

with gr.Blocks(
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="stone",
            neutral_hue="slate"
        ).set(
            body_background_fill="linear-gradient(135deg, #14283C, #B4A788)",
            body_text_color="#FFFFFF"
        ),
        title="🏡 AssistAura — Real Estate Chatbot"
) as demo:
    # Header - Minimal
    gr.HTML(
        """
        <div class="header">
            <h1>🏡 AssistAura</h1>
        </div>
        """,
        elem_classes="header"
    )

    with gr.Row(equal_height=False):
        # Chat area (left column)
        with gr.Column(scale=8, min_width=700):
            gr.Markdown("### 💬 Chat", elem_classes="section-title")
            chatbot = gr.Chatbot(
                type="messages",
                height=600,
                elem_classes="chatbot-container",
                show_copy_button=True,
                render_markdown=True,
                show_share_button=False
            )

            # Chat input with send button
            with gr.Row(elem_classes="chat-input-row"):
                msg = gr.Textbox(
                    placeholder="💡 Ask about apartments, villas, prices, locations in Egypt...",
                    show_label=False,
                    autofocus=True,
                    elem_classes="input-field",
                    lines=1,
                    max_lines=3,
                    container=False,
                    min_width=350
                )

            with gr.Row():
                with gr.Column(scale=1):
                    clear = gr.Button(
                        "🧹 Clear Chat",
                        variant="secondary",
                        elem_classes="button-secondary"
                    )
                with gr.Column(scale=2):
                    k_slider = gr.Slider(
                        1, 10,
                        value=5,
                        step=1,
                        label="📊 Number of Properties to Show",
                        interactive=True,
                        elem_classes="slider-container",
                        show_label=True,
                        minimum=1,
                        maximum=10,
                    )

        # Property results (right column)
        with gr.Column(scale=5, min_width=350):
            gr.Markdown("### 🔎 Relevant Properties", elem_classes="section-title")
            hits_display = gr.HTML(
                '<div class="no-results">✨ No results yet. Start asking about properties!</div>',
                elem_classes="results-container"
            )

    # State management
    session_state = gr.State(str(uuid.uuid4()))

    # Event handlers
    msg.submit(
        user_message,
        [msg, chatbot, session_state, k_slider],
        [chatbot, msg, session_state, hits_display],
        queue=True
    )

    clear.click(
        lambda: ([], str(uuid.uuid4()),
                 '<div class="no-results">✨ No results yet. Start asking about properties!</div>'),
        None,
        [chatbot, session_state, hits_display],
        queue=False
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )