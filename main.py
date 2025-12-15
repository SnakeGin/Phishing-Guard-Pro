from fastapi import UploadFile, File, FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from email import policy
from email.parser import BytesParser
from pydantic import BaseModel
from typing import Optional, List
import torch
import uvicorn
import httpx
import os
import json
import datetime
import re
from bs4 import BeautifulSoup # 引入 BS4 用于HTML取证分析

# --- 数据库相关导入 ---
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 导入核心模块
from feature_extractor import PhishingFeatureExtractor
from model_architecture import FMPEDModel

# ==========================================
# ⚙️ 配置区域
# ==========================================
LLM_API_KEY = "sk-genwnvgxggzilqhrkgmelgiylwskasedyemtzxadenqfgykx" 
LLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions" 
# LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
DATABASE_URL = "sqlite:///./phishing_logs.db" 

# --- 1. 数据库初始化 ---
Base = declarative_base()

class DetectionRecord(Base):
    __tablename__ = "records"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    risk_score = Column(Float)
    verdict = Column(String)
    risk_level = Column(String)
    features_json = Column(Text)
    defense_suggestion = Column(Text)
    email_content = Column(Text)
    forensic_data = Column(Text) # <--- [新增] 存储详细取证列表(JSON字符串)
    created_at = Column(DateTime, default=datetime.datetime.now)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. FastAPI 初始化 ---
app = FastAPI(title="MH-PDS Backend Pro", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_engine = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 数据模型 ---
class HistoryItem(BaseModel):
    id: int
    filename: str
    risk_score: float
    verdict: str
    risk_level: str
    created_at: str
    features_summary: dict
    email_content: Optional[str] = ""
    defense_suggestion: Optional[str] = ""
    forensic_data: List[str] = [] # <--- [新增] 返回给前端的证据列表

    class Config:
        orm_mode = True

class AnalysisResult(BaseModel):
    id: int
    risk_score: float
    verdict: str
    risk_level: str
    features_summary: dict
    defense_suggestion: str
    email_content: str
    forensic_data: List[str] # <--- [新增]
    processing_time: float

# --- 4. 生命周期 ---
@app.on_event("startup")
async def startup_event():
    print(f"🚀 系统启动中... 运行设备: {DEVICE}")
    model_engine['extractor'] = PhishingFeatureExtractor()
    try:
        fmped_model = FMPEDModel().to(DEVICE)
        fmped_model.load_state_dict(torch.load("fmped_model.pth", map_location=DEVICE))
        fmped_model.eval()
        model_engine['detector'] = fmped_model
        print("✅ 真实模型 (FMPED) 加载成功！")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e} (将使用演示逻辑)")
        model_engine['detector'] = None

# --- 5. 核心逻辑函数 ---

# [新增] 详细取证生成器
def generate_forensic_report(text_content, html_content):
    """
    扫描内容，提取具体的“罪证”字符串，用于前端展示
    """
    evidence = []
    
    # 1. 扫描敏感词 (提取上下文)
    # 定义高危词库
    keywords = ["立即", "24小时", "冻结", "suspend", "urgent", "verify", "login", "password", "bank", "refund", "帐户", "异常", "立即", "24小时", "冻结", "suspend", "urgent", "immediate", "breach","点击", "登录", "验证", "verify", "login", "click here", "update", "sign in","银行", "退税", "中奖", "bank", "refund", "invoice", "payment", "winner"]
    hits = []
    text_lower = text_content.lower()
    for kw in keywords:
        if kw in text_lower:
            hits.append(kw)
    if hits:
        # 去重并只取前5个
        unique_hits = list(set(hits))[:5]
        evidence.append(f"⚠️ 发现 {len(hits)} 个高危诱导词: {', '.join(unique_hits)}...")

    # 2. 扫描 IP 直连链接
    ip_pattern = re.compile(r'http[s]?://(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')
    ip_links = ip_pattern.findall(text_content)
    for ip in list(set(ip_links))[:3]: # 只展示前3个去重IP
        evidence.append(f"🚫 检测到裸 IP 链接 (绕过域名检测): {ip}")

    # 3. 扫描 HTML 特征
    if html_content:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 3.1 隐藏 iframe
            hidden_iframes = soup.find_all('iframe', attrs={"style": re.compile(r'display:\s*none|visibility:\s*hidden')})
            small_iframes = soup.find_all('iframe', width="0", height="0")
            if hidden_iframes or small_iframes:
                evidence.append("🕵️‍♂️ 检测到不可见 Iframe 标签 (可能包含 Drive-by Download 攻击)")
                
            # 3.2 密码表单
            password_inputs = soup.find_all('input', type='password')
            if password_inputs:
                evidence.append("🔓 检测到非法的密码收集表单 (Credential Harvesting)")
                    
            # 3.3 链接不一致 (Link Mismatch)
            # 查找所有链接，看文本和href是否差异巨大
            links = soup.find_all('a', href=True)
            for link in links:
                visible = link.get_text().strip()
                href = link['href']
                
                # 如果显示的文本像个域名，但href不包含它
                if re.match(r'^(http|www)', visible):
                    # 简单提取域名比较
                    visible_clean = visible.replace('https://', '').replace('http://', '').split('/')[0]
                    if len(visible_clean) > 5 and visible_clean not in href:
                        evidence.append(f"🎣 发现“表里不一”的欺诈链接: 显示 '{visible[:30]}' 但指向 '{href[:30]}...'")
                        break # 只报一个典型
        except:
            pass # HTML解析容错

    if not evidence:
        evidence.append("✅ 未检测到具体的硬特征指纹 (可能是纯语义攻击)")
        
    return evidence

async def generate_real_ai_advice(risk_score: float, details: dict, email_text: str) -> str:
    verdict = "高危钓鱼邮件" if risk_score > 75 else "可疑邮件" if risk_score > 45 else "安全邮件"
    email_snippet = email_text[:1000].replace('\n', ' ')
    
    system_prompt = "你是一个网络安全专家。请根据检测数据生成一份简短防御建议。不要包含Markdown标题(#)，直接分点说明风险和建议。"
    user_prompt = f"""
    结果: {verdict} (分值: {risk_score})
    特征: {json.dumps(details, ensure_ascii=False)}
    邮件: {email_snippet}...
    请给出约150字的分析与建议。
    """

    if "sk-xxx" in LLM_API_KEY:
        return generate_fallback_advice(risk_score, details)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            payload = {
                "model": LLM_MODEL_NAME,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": 0.3
            }
            response = await client.post(LLM_API_URL, json=payload, headers={"Authorization": f"Bearer {LLM_API_KEY}"})
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return generate_fallback_advice(risk_score, details)

def generate_fallback_advice(score, details):
    return "无法连接AI服务。建议：不要点击任何链接，向IT部门核实。"

def parse_eml_content(file_bytes: bytes):
    msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
    subject = msg.get('subject', '无主题')
    text_content = ""
    html_content = ""
    if msg.get_body(preferencelist=('plain')): text_content = msg.get_body(preferencelist=('plain')).get_content()
    if msg.get_body(preferencelist=('html')): html_content = msg.get_body(preferencelist=('html')).get_content()
    if not text_content and html_content:
        from bs4 import BeautifulSoup
        text_content = BeautifulSoup(html_content, "html.parser").get_text()
    if not text_content:
         for part in msg.walk():
            if part.get_content_type() == "text/plain": text_content += part.get_content()
    return subject, text_content, html_content

# --- 6. API 接口 ---

@app.get("/api/history", response_model=List[HistoryItem])
def read_history(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    records = db.query(DetectionRecord).order_by(DetectionRecord.created_at.desc()).offset(skip).limit(limit).all()
    results = []
    for r in records:
        item = HistoryItem(
            id=r.id,
            filename=r.filename,
            risk_score=r.risk_score,
            verdict=r.verdict,
            risk_level=r.risk_level,
            created_at=r.created_at.strftime("%Y-%m-%d %H:%M"),
            features_summary=json.loads(r.features_json) if r.features_json else {},
            email_content=r.email_content or "无内容",
            defense_suggestion=r.defense_suggestion or "暂无建议",
            # [新增] 解析 JSON 列表，如果为空则返回空列表
            forensic_data=json.loads(r.forensic_data) if r.forensic_data else [] 
        )
        results.append(item)
    return results

@app.post("/api/analyze-file", response_model=AnalysisResult)
async def analyze_email_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    import time
    start_time = time.time()
    
    content = await file.read()
    subject, text_content, html_content = parse_eml_content(content)
    
    extractor = model_engine['extractor']
    detector = model_engine['detector']
    
    html_input = html_content if html_content else text_content
    # 注意：特征提取需要两个参数，训练时如果只用了一个，这里要保持一致
    feature_data = extractor.process_email(text_content[:5000], html_input[:5000])
    
    # [新增] 生成详细取证数据
    forensic_evidence = generate_forensic_report(text_content, html_input)

    risk_score = 0.0
    if detector:
        input_tensor = torch.tensor(feature_data['fused_vector'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw_score = detector(input_tensor).item() * 100
            
            # 权重修正逻辑
            feats = feature_data['details']
            hard_evidence_count = feats.get('url_count', 0) + (1 if feats.get('has_iframe') else 0) + (1 if feats.get('suspicious_ip_urls', 0) > 0 else 0)
            
            if hard_evidence_count == 0:
                risk_score = raw_score * 0.6 
            else:
                risk_score = raw_score
    else:
        risk_score = 88.5

    if risk_score > 75: verdict, risk_level = "Phishing", "High"
    elif risk_score > 45: verdict, risk_level = "Suspicious", "Medium"
    else: verdict, risk_level = "Safe", "Low"
    
    suggestion = await generate_real_ai_advice(risk_score, feature_data['details'], text_content)
    
    db_record = DetectionRecord(
        filename=file.filename,
        risk_score=risk_score,
        verdict=verdict,
        risk_level=risk_level,
        features_json=json.dumps(feature_data['details']),
        defense_suggestion=suggestion,
        email_content=html_input,
        forensic_data=json.dumps(forensic_evidence) # [新增] 存入数据库
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    process_time = time.time() - start_time
    
    return {
        "id": db_record.id,
        "risk_score": round(risk_score, 2),
        "verdict": verdict,
        "risk_level": risk_level,
        "features_summary": feature_data['details'],
        "defense_suggestion": suggestion,
        "email_content": html_input,
        "forensic_data": forensic_evidence, # [新增] 返回给前端
        "processing_time": round(process_time, 3)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)