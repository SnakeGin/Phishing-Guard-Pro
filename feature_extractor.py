import torch
import numpy as np
import re
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertModel
from urllib.parse import urlparse
import warnings

warnings.filterwarnings("ignore")

class PhishingFeatureExtractor:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        print(f"正在加载 BERT 模型: {model_name} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name).to(self.device)
        self.bert_model.eval()
        
        # 升级版敏感词库：分类更细
        self.keywords_urgent = ["立即", "24小时", "冻结", "suspend", "urgent", "immediate", "breach"]
        self.keywords_action = ["点击", "登录", "验证", "verify", "login", "click here", "update", "sign in"]
        self.keywords_financial = ["银行", "退税", "中奖", "bank", "refund", "invoice", "payment", "winner"]
        
    def _analyze_dom_structure(self, html_content):
        """
        深度 DOM 分析 (针对 HTML 钓鱼手法)
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 1. 检测“表里不一”的链接 (Link Mismatch) - 这是最明显的钓鱼特征
        # 文字显示是 google.com，实际指向 evil.com
        mismatch_count = 0
        links = soup.find_all('a', href=True)
        for link in links:
            visible_text = link.get_text().strip().lower()
            actual_href = link['href'].lower()
            
            # 如果可见文本看起来像个域名/URL，但和实际 href 不匹配
            if re.match(r'^(http|www|[a-z0-9-]+\.[a-z]+)', visible_text):
                # 简单过滤掉 http://google.com 和 google.com 的差异
                clean_text = visible_text.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0]
                clean_href = actual_href.replace("https://", "").replace("http://", "").replace("www.", "").split('/')[0]
                
                if clean_text not in clean_href and len(clean_text) > 4:
                    mismatch_count += 1

        # 2. 检测密码表单 (Credential Harvesting)
        # 很多钓鱼邮件会伪造一个登录框
        has_password_field = 0
        inputs = soup.find_all('input')
        for inp in inputs:
            if inp.get('type') == 'password' or 'pass' in inp.get('name', '').lower():
                has_password_field = 1
                break

        # 3. 检测 iframe 和 script
        iframe_count = len(soup.find_all('iframe'))
        script_count = len(soup.find_all('script'))
        
        return [mismatch_count, has_password_field, iframe_count, script_count]

    def _analyze_url_forensics(self, text):
        """
        深度 URL 取证 (针对 URL 混淆)
        """
        # 提取所有 URL
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        urls = url_pattern.findall(text)
        
        suspicious_ip_urls = 0
        at_symbol_redirect = 0 # 检测 http://google.com@evil.com
        multiple_subdomains = 0 # 检测 http://paypal.com.verify.login.evil.com
        short_link_count = 0
        
        short_domains = ['bit.ly', 'goo.gl', 't.cn', 'tinyurl', 'is.gd']
        ip_pattern = re.compile(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})')

        for url in urls:
            # 1. IP 地址检测 (包含 192.168... 或 0x12 这种混淆)
            if ip_pattern.search(url):
                suspicious_ip_urls += 1
            
            # 2. @ 符号重定向检测 (浏览器会忽略 @ 前面的内容，直接访问后面)
            if '@' in url:
                at_symbol_redirect += 1
                
            # 3. 短链检测
            if any(s in url for s in short_domains):
                short_link_count += 1
                
            # 4. 多级子域名检测 (超过4级通常可疑)
            try:
                parsed = urlparse(url)
                if parsed.hostname and len(parsed.hostname.split('.')) > 4:
                    multiple_subdomains += 1
            except:
                pass
                
        return [len(urls), suspicious_ip_urls, at_symbol_redirect, short_link_count]

    def _get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

    def process_email(self, raw_text, html_content=""):
        """
        特征融合入口
        """
        # 确保 html_content 存在，如果不存在则尝试用 text 模拟
        dom_content = html_content if html_content and len(html_content) > 10 else raw_text
        
        # 1. 深度 DOM 分析 (4维)
        # [LinkMismatch, PasswordField, Iframe, Script]
        feat_dom = self._analyze_dom_structure(dom_content)
        
        # 2. 深度 URL 分析 (4维)
        # [Total, IP, @Redirect, ShortLink]
        feat_url = self._analyze_url_forensics(raw_text)
        
        # 3. 关键词分析 (1维)
        count = 0
        text_lower = raw_text.lower()
        for kw in self.keywords_urgent + self.keywords_action + self.keywords_financial:
            if kw in text_lower:
                count += 1
        feat_kw = [count]
        
        # 传统特征向量 (9维) -> 这必须和模型输入的 traditional_dim 一致！
        # 如果你改变了这里的维度，必须重新修改模型定义并重新训练！
        # 为了兼容之前的代码，我们保持 9 维，但内容换成了更强的特征：
        # 1. Mismatch (新!)
        # 2. PasswordField (新!)
        # 3. Iframe
        # 4. Script
        # 5. URL Total
        # 6. IP URL
        # 7. @ Redirect (新!)
        # 8. Short Link
        # 9. Keywords
        
        traditional_features = np.array(feat_dom + feat_url + feat_kw, dtype=np.float32)
        
        # BERT 特征
        bert_features = self._get_bert_embedding(raw_text)
        fused_vector = np.concatenate((traditional_features, bert_features))
        
        return {
            "fused_vector": fused_vector,
            "details": {
                # 这些用于前端展示和逻辑判断
                "link_mismatch": feat_dom[0],       # 极高危
                "has_password_form": feat_dom[1] > 0, # 极高危
                "has_iframe": feat_dom[2] > 0,
                "url_count": feat_url[0],
                "suspicious_ip_urls": feat_url[1],
                "at_redirect": feat_url[2],         # 高危
                "keyword_hits": feat_kw[0]
            }
        }