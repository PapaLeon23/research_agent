import streamlit as st
import os
import re
import requests
import json
import time
from fpdf import FPDF
from typing import List, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from youtube_transcript_api import YouTubeTranscriptApi
from langgraph.graph import StateGraph, END

# --- 1. API 키 설정 ---
ANT_KEY = st.secrets.get("ANTHROPIC_API_KEY")
GEM_KEY = st.secrets.get("GOOGLE_API_KEY")
TAV_KEY = st.secrets.get("TAVILY_API_KEY")
GAMMA_KEY = st.secrets.get("GAMMA_API_KEY")

# --- 2. 초기 세션 상태 ---
if "history" not in st.session_state:
    st.session_state.history = []
if "selected_report" not in st.session_state:
    st.session_state.selected_report = None

# --- 3. 에이전트 핵심 로직 ---
class AgentState(TypedDict):
    topic: str
    plan: List[str]
    context: List[str]
    iteration: int
    next_step: str
    final_title: str # AI가 생성할 최종 제목 추가

def create_agent():
    fast_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=GEM_KEY)
    smart_llm = ChatAnthropic(model="claude-sonnet-4-6", anthropic_api_key=ANT_KEY)
    search_tool = TavilySearchResults(k=5, tavily_api_key=TAV_KEY)

    def planner_node(state: AgentState):
        st.write("🎯 **Step 1. Planner**: 리서치 전략 수립 중...")
        prompt = f"'{state['topic']}'에 대해 2026년 최신 정보를 찾기 위한 검색 키워드 3개를 한 줄에 하나씩만 출력해줘."
        res = fast_llm.invoke(prompt)
        raw_res = res.content if hasattr(res, 'content') else res
        if isinstance(raw_res, list):
            raw_text = " ".join([str(item.get('text', item) if isinstance(item, dict) else item) for item in raw_res])
        else: raw_text = str(raw_res)
        queries = [re.sub(r'^[0-9.\-\*\s]+', '', q).strip() for q in raw_text.split('\n') if q.strip()][:3]
        return {"plan": queries or [state['topic']], "iteration": 0, "context": []}

    def searcher_node(state: AgentState):
        query = state['plan'][state['iteration'] % len(state['plan'])]
        st.write(f"🔍 **Step 2. Searcher**: '{query}' 데이터 수집 중...")
        try:
            results = search_tool.invoke({"query": query})
            formatted = [f"소스: {r['url']}\n내용: {r['content']}" for r in results if isinstance(r, dict)]
            return {"context": state['context'] + (formatted if formatted else [f"'{query}' 결과 없음"])}
        except: return {"context": state['context'] + ["검색 오류"]}

    def youtube_node(state: AgentState):
        st.write("📺 **Step 3. YouTube**: 영상 분석 중...")
        yt_summary = "관련 영상 없음"
        full_ctx = "".join(state['context'])
        match = re.search(r'(https?://(?:www\.)?youtube\.com/watch\?v=[0-9A-Za-z_-]{11}|https?://youtu\.be/[0-9A-Za-z_-]{11})', full_ctx)
        if match:
            url = match.group(1)
            vid_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', url)
            if vid_match:
                video_id = vid_match.group(1)
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
                    yt_summary = f"유튜브({url}): " + " ".join([t['text'] for t in transcript])[:1000]
                except: pass
        return {"context": state['context'] + [yt_summary]}

    def critic_node(state: AgentState):
        st.write("⚖️ **Step 4. Critic**: 정보 충분성 검토 중...")
        prompt = f"자료가 충분하면 'FINISH', 부족하면 'MORE / 사유'라고 답하세요.\n자료: {state['context']}"
        res = smart_llm.invoke(prompt)
        resp = str(res.content)
        if "MORE" in resp and state['iteration'] < 2:
            return {"next_step": "continue", "iteration": state['iteration'] + 1}
        return {"next_step": "end", "iteration": state['iteration'] + 1}

    def synthesizer_node(state: AgentState):
        st.write("📝 **Step 5. Synthesizer**: 최종 리포트 및 제목 생성 중...")
        
        # 1. 전문적인 제목 생성
        title_prompt = f"사용자 질문: {state['topic']}\n수집 자료 요약: {str(state['context'])[:500]}\n위 내용을 바탕으로 보고서에 어울리는 전문적이고 간결한 제목(한 줄)을 생성하세요."
        title_res = smart_llm.invoke(title_prompt)
        final_title = str(title_res.content).strip().replace('"', '').replace("'", "")

        # 2. 보고서 본문 작성 (참고 자료 URL 포함 지시 강화)
        report_prompt = f"""
        당신은 전문 리서치 애널리스트입니다. 아래 자료를 바탕으로 고퀄리티 리포트를 작성하세요.
        
        제목: {final_title}
        자료: {state['context']}
        
        [작성 가이드라인]
        1. 자료에 포함된 팩트를 중심으로 심층 분석 내용을 작성하세요.
        2. 반드시 보고서 마지막 섹션으로 **[주요 참고 출처]**를 만드세요.
        3. 수집된 자료(context)에 포함된 원문 URL 주소들을 추출하여 웹 사이트와 영상 소스별로 리스트업하세요.
        4. 웹 사이트 URL은 최소 3개 이상 포함하세요.
        5. 마지막 워터마크는 'Joosung's Agent Report'를 넣으세요.
        """
        report_res = smart_llm.invoke(report_prompt)
        
        return {"context": [report_res.content], "final_title": final_title}

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node); workflow.add_node("searcher", searcher_node)
    workflow.add_node("youtube", youtube_node); workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "searcher"); workflow.add_edge("searcher", "youtube")
    workflow.add_edge("youtube", "critic")
    workflow.add_conditional_edges("critic", lambda x: x["next_step"], {"continue": "searcher", "end": "synthesizer"})
    workflow.add_edge("synthesizer", END)
    return workflow.compile()

# --- 4. 유틸리티 함수 ---
def create_gamma_presentation(topic, report_content, style_instruction):
    if not GAMMA_KEY: return None, "API 키 누락"
    BASE_URL = "https://public-api.gamma.app/v1.0/generations"
    headers = {"X-API-KEY": GAMMA_KEY, "Content-Type": "application/json", "accept": "application/json"}
    payload = {
        "inputText": f"Topic: {topic}\n\nContent: {report_content[:5000]}\n\nStyle Guide: {style_instruction}",
        "textMode": "preserve", "format": "presentation", "exportAs": "pptx", "textOptions": {"language": "ko"}
    }
    try:
        response = requests.post(BASE_URL, json=payload, headers=headers)
        if response.status_code not in [200, 201]: return None, f"Gamma 에러: {response.status_code}"
        gen_id = response.json().get("generationId")
        for _ in range(30):
            time.sleep(10)
            status_res = requests.get(f"{BASE_URL}/{gen_id}", headers=headers)
            status_data = status_res.json()
            if status_data.get("status") == "completed": return status_data.get("gammaUrl"), "성공"
            elif status_data.get("status") == "error": return None, "Gamma 내부 오류"
        return None, "시간 초과"
    except Exception as e: return None, str(e)

def create_professional_pdf(text, title):
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
    pdf = FPDF()
    pdf.add_page()
    
    # 폰트 경로 설정
    eb_font = "NanumSquareEB.ttf"
    r_font = "NanumSquareR.ttf"
    
    # 제목용 폰트 (Extra Bold) 로드
    if os.path.exists(eb_font):
        pdf.add_font('NS_EB', '', eb_font)
        title_font = 'NS_EB'
    else:
        title_font = 'Arial'
        st.error("NanumSquareEB.ttf 파일을 찾을 수 없습니다.")

    # 본문용 폰트 (Regular) 로드
    if os.path.exists(r_font):
        pdf.add_font('NS_R', '', r_font)
        body_font = 'NS_R'
    else:
        body_font = 'Arial'

    # 1. 제목 섹션 (EB 적용)
    pdf.set_font(title_font, size=22)
    pdf.set_text_color(31, 73, 125) # 다크 블루
    pdf.multi_cell(0, 15, txt=title, align='L')
    pdf.line(10, pdf.get_y()+2, 200, pdf.get_y()+2)
    pdf.ln(10)
    
    # 2. 본문 섹션 (R 적용)
    pdf.set_font(body_font, size=11)
    pdf.set_text_color(0)
    clean_text = text.replace('#', '').replace('*', '').replace('>', '').replace('•', '-')
    pdf.multi_cell(0, 8, txt=clean_text)
    
    # 3. 푸터
    pdf.set_y(-15)
    pdf.set_font(body_font, size=8)
    pdf.set_text_color(150)
    pdf.cell(0, 10, f'Joosung\'s Agent Report | {title}', 0, 0, 'C')
    
    return bytes(pdf.output()), safe_title

# --- 5. UI 영역 ---
st.set_page_config(page_title="2026 AI Research Studio", layout="wide")

with st.sidebar:
    st.title("📂 History")
    if st.button("➕ 새 리서치 시작", use_container_width=True):
        st.session_state.selected_report = None
        st.rerun()
    st.divider()
    for i, entry in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            # 리스트에 AI가 생성한 제목 표시
            if st.button(f"📄 {entry['title'][:15]}...", key=f"h_{idx}", use_container_width=True):
                st.session_state.selected_report = entry
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.history.pop(idx)
                if st.session_state.selected_report == entry: st.session_state.selected_report = None
                st.rerun()

if st.session_state.selected_report:
    item = st.session_state.selected_report
    st.header(f"📑 {item['title']}")
    st.markdown(item['report'])
    pdf_bytes, safe_filename = create_professional_pdf(item['report'], item['title'])
    st.download_button("📩 PDF 보고서 다운로드", data=pdf_bytes, file_name=f"{safe_filename}.pdf")
    
    st.divider()
    st.subheader("🎨 Gamma AI PPT 제작")
    style_input = st.text_input("디자인 컨셉 입력", placeholder="예: 깔끔한 화이트 톤의 비즈니스 스타일")
    if st.button("🚀 Gamma PPT 생성"):
        with st.status("🪄 생성 중...") as s:
            url, msg = create_gamma_presentation(item['title'], item['report'], style_input)
            if url:
                s.update(label="✅ 완성!", state="complete")
                st.components.v1.html(f"<script>window.open('{url}', '_blank');</script>", height=0)
                st.link_button("📊 PPT 수동 열기", url)
            else: st.error(msg)

query = st.chat_input("조사할 주제를 입력하세요...")
if query:
    with st.chat_message("user"): st.markdown(query)
    with st.chat_message("assistant"):
        with st.status("🕵️ 에이전트 가동 중...", expanded=True) as status:
            agent = create_agent()
            result = agent.invoke({"topic": query, "context": [], "iteration": 0})
            status.update(label="✅ 리서치 완료!", state="complete")
        
        final_report = result['context'][-1]
        final_title = result['final_title'] # AI가 생성한 제목
        
        st.subheader(f"📑 {final_title}")
        st.markdown(final_report)
        
        st.session_state.history.append({"title": final_title, "report": final_report})
        st.session_state.selected_report = {"title": final_title, "report": final_report}

        st.rerun()

