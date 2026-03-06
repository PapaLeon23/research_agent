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
MANUS_API_KEY = st.secrets.get("MANUS_API_KEY")

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
    final_title: str 

def create_agent():
    fast_llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=GEM_KEY)
    smart_llm = ChatAnthropic(model="claude-sonnet-4-6", anthropic_api_key=ANT_KEY)
    search_tool = TavilySearchResults(k=5, tavily_api_key=TAV_KEY)

    def planner_node(state: AgentState):
        st.write("🎯 **Step 1. Planner**: 리서치 전략 수립 중...")
        prompt = f"'{state['topic']}'에 대해 2026년 최신 정보를 찾기 위한 검색 키워드 3개를 한 줄에 하나씩만 출력해줘."
        res = fast_llm.invoke(prompt)
        
        # [노이즈 제거] 객체 형태가 아닌 순수 텍스트만 추출하도록 로직 강화
        if hasattr(res, 'content'):
            raw_text = str(res.content)
        else:
            raw_text = str(res)
        
        # JSON 형태나 dict 형태의 문자열이 섞여 들어오는 경우 처리
        if "text': '" in raw_text:
            match = re.search(r"text':\s*'([^']*)'", raw_text)
            if match: raw_text = match.group(1)

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
        prompt = f"""
        현재까지 수집된 자료가 '{state['topic']}'에 대해 충분한지 판단하세요.
        자료: {state['context']}
        
        결과가 충분하면 'FINISH'라고 답하세요.
        부족하다면 반드시 'MORE / 부족한 사유(한 문장)' 형식으로 답하세요.
        """
        res = smart_llm.invoke(prompt)
        resp = str(res.content).strip()
        
        if "MORE" in resp and state['iteration'] < 2:
            # "MORE / 사유"에서 사유 부분만 추출하여 화면에 표시
            reason = resp.split('/')[-1].strip() if '/' in resp else "추가적인 세부 정보 수집이 필요합니다."
            st.warning(f"🔍 **정보 보완 필요**: {reason}")
            return {"next_step": "continue", "iteration": state['iteration'] + 1}
        
        st.success("✅ 정보가 충분합니다. 최종 보고서를 작성합니다.")
        return {"next_step": "end", "iteration": state['iteration'] + 1}

    def synthesizer_node(state: AgentState):
        st.write("📝 **Step 5. Synthesizer**: 최종 보고서 및 제목 생성 중...")
        
        # 1. 전문적인 제목 생성
        title_prompt = f"사용자 질문: {state['topic']}\n수집 자료 요약: {str(state['context'])[:500]}\n위 내용을 바탕으로 보고서에 어울리는 전문적이고 간결한 제목(한 줄)을 생성하세요."
        title_res = smart_llm.invoke(title_prompt)
        final_title = str(title_res.content).strip().replace('"', '').replace("'", "")

        # 2. 보고서 본문 작성
        report_prompt = f"제목: {final_title}\n자료: {state['context']}\n2026년 기준 전문 리포트를 작성하세요. 마지막엔 'Joosung's Agent Report'를 넣으세요."
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
def create_manus_infographic(topic, report_content, style_instruction):
    if not MANUS_API_KEY: return None, "Manus API 키 누락"
    url = "https://api.manus.ai/v1/tasks"
    headers = {"API_KEY": MANUS_API_KEY, "Content-Type": "application/json"}
    data = {
        "prompt": f"Create a professional 16:9 infographic presentation.\nTopic: {topic}\nStyle: {style_instruction}\nContent: {report_content[:4000]}",
        "agentProfile": "manus-1.6-lite",
        "taskMode": "agent",
        "createShareableLink": True
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code not in [200, 201]: return None, "API 연결 실패"
        task_id = response.json().get("task_id")
        with st.status("📊 Manus 디자인 제작 중...", expanded=True) as s:
            for _ in range(60):
                time.sleep(10)
                res = requests.get(f"{url}/{task_id}", headers={"API_KEY": MANUS_API_KEY}).json()
                if res.get("status") == "completed":
                    files = res.get("files", [])
                    pptx_url = next((f['url'] for f in files if f.get('filename', '').endswith('.pptx')), None)
                    return pptx_url or res.get("share_url"), "성공"
        return None, "시간 초과"
    except Exception as e: return None, str(e)

def create_professional_pdf(text, title):
    pdf = FPDF(); pdf.add_page()
    eb_font, r_font = "NanumSquareEB.ttf", "NanumSquareR.ttf"
    if os.path.exists(eb_font): pdf.add_font('NS_EB', '', eb_font); t_f = 'NS_EB'
    else: t_f = 'Arial'
    if os.path.exists(r_font): pdf.add_font('NS_R', '', r_font); b_f = 'NS_R'
    else: b_f = 'Arial'
    pdf.set_font(t_f, size=20); pdf.cell(0, 15, txt=title, ln=1, align='L')
    pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(10)
    pdf.set_font(b_f, size=11); pdf.multi_cell(0, 8, txt=text.replace('#', '').replace('*', ''))
    return bytes(pdf.output()), re.sub(r'[\\/*?:"<>|]', "", title)

# --- 5. UI 영역 ---
st.set_page_config(page_title="2026 Business Research Studio", layout="wide")

# 사이드바: 히스토리 관리
with st.sidebar:
    st.title("💼 Research Archive")
    if st.button("➕ New Research", use_container_width=True):
        st.session_state.selected_report = None
        st.rerun()
    st.divider()
    for i, entry in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - 1 - i
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(f"📄 {entry['title'][:15]}", key=f"h_{idx}", use_container_width=True):
                st.session_state.selected_report = entry
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.history.pop(idx); st.rerun()

# 메인 화면 UI
if not st.session_state.selected_report:
    # 썰렁하지 않게 메인 대시보드 꾸미기
    st.title("🚀 2026 AI Business Research Studio")
    st.subheader("전문적인 비즈니스 인텔리전스를 위한 에이전트 기반 리서치 도구")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🔍 **Deep Search**\nTavily & YouTube를 통한 실시간 데이터 수집")
    with col2:
        st.info("⚖️ **Auto Critic**\nClaude 3.5 & Gemini 기반 정보 충분성 검토")
    with col3:
        st.info("📊 **Manus Design**\n인포그래픽 슬라이드 자동 생성")
    
    st.divider()
    st.write("#### 💡 리서치 가이드")
    st.markdown("""
    1. 하단 채팅창에 **조사하고 싶은 비즈니스 주제**를 입력하세요.
    2. 에이전트가 5단계(계획-웹 탑색-영상 탐색-점검-생성성) 공정을 시작합니다.
    3. 결과가 나오면 **PDF 보고서** 다운로드 및 **Manus 인포그래픽** 제작이 가능합니다.
    """)
else:
    # 1. [질문 상단 고정] 사용자가 무엇을 물었는지 항상 보이도록 설정
    item = st.session_state.selected_report
    
    # 상단 헤더 스타일링
    st.info(f"🔍 **현재 분석 리포트:** {item['title']}")
    
    # 2. [리포트 본문] 탭을 제거하고 전체 화면에 바로 노출
    st.markdown("---")
    st.markdown(item['report'])
    
    # 3. [하단 액션 영역] 보고서가 끝나는 지점에 버튼 배치
    st.divider()
    st.subheader("🛠️ 리포트 후속 작업")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("📂 **문서 저장**")
        pdf_bytes, safe_name = create_professional_pdf(item['report'], item['title'])
        st.download_button(
            "📩 PDF 보고서 다운로드", 
            data=pdf_bytes, 
            file_name=f"{safe_name}.pdf",
            use_container_width=True
        )
    
    with col2:
        st.write("🎨 **슬라이드 작성**")
        # Manus API 키 재확인 로직 (인식 오류 방지)
        current_manus_key = st.secrets.get("MANUS_API_KEY")
        
        with st.expander("📊 Manus 인포그래픽 슬라이드 생성", expanded=True):
            style_input = st.text_input(
                "디자인 테마", 
                placeholder="예: 'Professional Blue, Modern, High-tech'",
                key="manus_style_input"
            )
            
            if st.button("🚀 슬라이드 생성 시작", use_container_width=True):
                if not current_manus_key:
                    st.error("🚨 MANUS_API_KEY가 시스템에 등록되지 않았습니다. Secrets 설정을 확인해주세요.")
                else:
                    with st.spinner("Manus 에이전트가 슬라이드를 디자인하고 있습니다..."):
                        # 여기서 전역 변수 대신 직접 키를 넘기거나 함수 내에서 st.secrets를 쓰도록 되어있어야 합니다.
                        url, msg = create_manus_infographic(item['title'], item['report'], style_input)
                        if url:
                            st.success("✅ 인포그래픽 슬라이드 제작 완료!")
                            st.link_button("📂 결과물 확인 및 다운로드", url, use_container_width=True)
                        else:
                            st.error(f"❌ 생성 실패: {msg}")
                            
# 입력창 (고정)
query = st.chat_input("조사할 업무 주제를 입력하세요 (예: 2026년 eSIM 시장 전망)")
if query:
    with st.status("🕵️ 에이전트 리서치 가동 중...", expanded=True) as status:
        agent = create_agent()
        result = agent.invoke({"topic": query, "context": [], "iteration": 0})
        status.update(label="✅ 리서치 완료!", state="complete")
    
    final_report = result['context'][-1]
    final_title = result['final_title']
    st.session_state.history.append({"title": final_title, "report": final_report})
    st.session_state.selected_report = {"title": final_title, "report": final_report}

    st.rerun()
