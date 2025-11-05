"""
루팡이 카드 상담 챗봇 - Gradio 웹 인터페이스
"""
import gradio as gr
import pandas as pd
import os
import sys
import time
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.chatbot import FreeCardConsultingChatbot
from src.config import CSV_FILE


def img_to_base64(path):
    """이미지를 base64로 인코딩"""
    if not os.path.exists(path):
        print(f"경고: 이미지 파일을 찾을 수 없습니다: {path}")
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "lupanglogo.png")
ICON_PATH = os.path.join(ASSETS_DIR, "lupangg.png")

LOGO = img_to_base64(LOGO_PATH)
ICON = img_to_base64(ICON_PATH)


print("\n" + "="*70)
print("루팡이 카드 상담 챗봇 - Gradio 웹 인터페이스 시작")
print("="*70 + "\n")

try:
    df = pd.read_csv(CSV_FILE)
    print(f"데이터 파일 로드 완료: {CSV_FILE}\n")
    
    if "consulting_content" not in df.columns and "content" in df.columns:
        df["consulting_content"] = df["content"]
    
except FileNotFoundError:
    print(f"오류: 데이터 파일을 찾을 수 없습니다: {CSV_FILE}")
    print("data/ 폴더에 final_rag_data.csv 파일이 있는지 확인하세요.")
    exit(1)
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    exit(1)

try:
    chatbot = FreeCardConsultingChatbot(df)
except Exception as e:
    print(f"챗봇 초기화 중 오류 발생: {e}")
    exit(1)


def chat_fn(message, history):
    """Gradio 채팅 함수 (아이콘 외부 + 대화 누적)"""
    if not message.strip():
        return history, ""

    user_html = f"<div class='user-line'><div class='user-bubble'>{message}</div></div>"
    history.append((user_html, None))
    yield history, ""

    history.append((None, "<div class='bot-wrapper'><div class='bot-icon'></div><div class='bot-bubble'>답변을 준비 중입니다...</div></div>"))
    yield history, ""
    time.sleep(1)

    try:
        result = chatbot.chat(message)
        bot_reply = result.get("answer", "응답 생성 실패")
    except Exception as e:
        bot_reply = f"오류 발생: {e}"

    history[-1] = (
        None,
        f"<div class='bot-wrapper'><div class='bot-icon'></div><div class='bot-bubble'>{bot_reply}</div></div>",
    )
    yield history, ""


custom_css = f"""
.gradio-container {{
    background-color: #fafafa !important;
    font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
}}
#logo {{
    display: flex;
    justify-content: flex-start;
    align-items: center;
    margin-top: 10px;
    margin-left: 10px;
}}
#logo img {{
    height: 80px;
}}

.chat-frame {{
    display: flex;
    flex-direction: column;
    background-color: white;
    border-radius: 20px;
    overflow: hidden;
    height: 620px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #e0e0e0;
}}

.wrap, .wrap.svelte-1x0d2t9, .message, .chatbot > div {{
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}

.user-line {{
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
}}
.user-bubble {{
    background-color: #FFD9EC;
    color: #333;
    border-radius: 18px 18px 0 18px;
    padding: 10px 14px;
    max-width: 75%;
    word-break: keep-all;
    text-align: left;
    line-height: 1.4;
}}

.bot-wrapper {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 6px 0;
}}
.bot-icon {{
    width: 42px;
    height: 42px;
    background-image: url('data:image/png;base64,{ICON}');
    background-size: cover;
    background-position: center;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 2px;
}}
.bot-bubble {{
    background-color: #EAEAEA;
    color: #333;
    border-radius: 18px 18px 18px 0;
    padding: 10px 14px;
    line-height: 1.5;
    max-width: 80%;
}}

.input-area {{
    background-color: #fafafa;
    padding: 12px 16px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 10px;
}}
.input-area textarea {{
    resize: none !important;
    overflow: hidden !important;
    height: 36px !important;
    border: none !important;
    outline: none !important;
    background: transparent !important;
}}
.input-area button {{
    background-color: #FFD9EC !important;
    color: #4C4C4C !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 4px 12px !important;
    transition: 0.2s;
}}
.input-area button:hover {{
    background-color: #FFE6F2 !important;
}}
footer {{ display: none !important; }}
"""


with gr.Blocks(css=custom_css) as demo:
    if LOGO:
        gr.HTML(f"<div id='logo'><img src='data:image/png;base64,{LOGO}'></div>")
    else:
        gr.HTML("<div id='logo'><h2>루팡이 카드 상담 챗봇</h2></div>")
    
    gr.Markdown("> 당신의 시간을 지켜주는 금융 파트너")

    with gr.Column(elem_classes="chat-frame"):
        chatbot_ui = gr.Chatbot(height=520, sanitize_html=False)
        with gr.Row(elem_classes="input-area"):
            txt = gr.Textbox(
                placeholder="궁금한 금융 상담 내용을 입력하세요...",
                scale=9,
                lines=1,
                show_label=False,
                container=False,
            )
            btn = gr.Button("보내기", scale=1)

    txt.submit(chat_fn, [txt, chatbot_ui], [chatbot_ui, txt], queue=True)
    btn.click(chat_fn, [txt, chatbot_ui], [chatbot_ui, txt], queue=True)


if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
