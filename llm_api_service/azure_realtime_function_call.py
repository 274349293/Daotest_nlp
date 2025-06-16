import json
import re
import math
import asyncio
import time
import aiohttp
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
import jieba
import jieba.analyse
from collections import Counter

logger = CustomLogger(name="DaoTest optimized retrieval api", write_to_file=True)
"""
Azure realtime 实时语音模型 function call 调用结果返回接口
1.目前测试阶段支持两个topic：企业出海 和 汇仁培训
2.新增config，所有配置文件均可在 utils/prompt.json文件中配置

update:
1.250519 新增新闻检索接口，触发后调用news.api去搜寻相关新闻，然后做summary返回
2.优化返回格式，统一使用固定的result字段作为主要内容返回

"""
# 新闻API配置
NEWS_API_URL = "https://eventregistry.org/api/v1/article/getArticles"
NEWS_API_KEY = "81299327-0db9-44f5-8c7c-f879c302fe8b"

# 全局模型缓存 - 避免重复加载
_GLOBAL_MODEL_CACHE = {}
# 全局索引缓存 - 新增：缓存已构建的索引
_GLOBAL_INDEX_CACHE = {}
# 全局变量控制jieba加载
_JIEBA_LOADED = False


# 预加载 只有首次启动耗时
def preload_jieba():
    """预加载jieba分词库"""
    try:
        import jieba
        import jieba.analyse
        # 触发字典加载
        jieba.lcut("预加载测试")
        jieba.analyse.extract_tags("预加载测试", topK=1)
        logger.info("Jieba preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload jieba: {e}")


def get_cached_model(model_type: str, model_name: str = None):
    """获取缓存的模型"""
    cache_key = f"{model_type}_{model_name}" if model_name else model_type

    if cache_key not in _GLOBAL_MODEL_CACHE:
        try:
            if model_type == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("./temp_model")
                _GLOBAL_MODEL_CACHE[cache_key] = model
                logger.info(f"Loaded and cached {model_type} model")

            elif model_type == "reranker":
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                model.eval()
                _GLOBAL_MODEL_CACHE[f"{cache_key}_tokenizer"] = tokenizer
                _GLOBAL_MODEL_CACHE[f"{cache_key}_model"] = model
                logger.info(f"Loaded and cached reranker model")

        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            _GLOBAL_MODEL_CACHE[cache_key] = None

    return _GLOBAL_MODEL_CACHE.get(cache_key)


class RetrievalConfig(BaseModel):
    """检索配置"""
    # 关键词匹配配置
    enable_keyword_mapping: bool = True

    # 查询理解配置
    enable_query_understanding: bool = True
    enable_query_expansion: bool = True
    enable_intent_classification: bool = True
    enable_entity_extraction: bool = True

    # 检索策略配置
    enable_sparse_retrieval: bool = True
    enable_dense_retrieval: bool = False
    enable_reranking: bool = False

    # 失败回退策略
    fallback_strategy: str = "full_content"

    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    sparse_weight: float = 1.0
    dense_weight: float = 0.0
    rerank_weight: float = 0.0

    top_k_candidates: int = 10
    final_top_k: int = 5

    # ===== 新增：相似度阈值配置 =====
    sparse_min_score: float = 1.0  # BM25最低分数阈值
    dense_min_score: float = 0.5  # 密集检索最低相似度阈值（从0.3提高到0.5）
    keyword_min_relevance: float = 0.5  # 关键词相关性最低阈值

    # 性能优化配置
    enable_async: bool = True
    max_content_length: int = 1000
    enable_caching: bool = True


class QueryUnderstanding:
    """查询理解层 - 轻量化版本"""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        # 简化的意图关键词
        self.intent_keywords = {
            "操作指南": ["如何", "怎么", "方法", "步骤"],
            "时机判断": ["什么时候", "何时", "时机", "时间"],
            "可行性": ["要不要", "是否", "适合", "可以", "能否", "应该"]
        }

        # 简化的同义词
        self.domain_synonyms = {
            "出海": ["海外", "国际", "全球", "跨境"],
            "企业": ["公司", "集团", "组织"],
            "风险": ["危险", "威胁", "挑战", "问题"]
        }

    async def process_query(self, query: str) -> Dict[str, Any]:
        """快速查询处理"""
        start_time = time.time()

        result = {
            "original": query,
            "cleaned": self._clean_query(query),
            "intent": "课程培训提问",
            "entities": [],
            "expanded_query": query,
            "keywords": []
        }

        if self.config.enable_intent_classification:
            result["intent"] = self._classify_intent(query)

        if self.config.enable_query_expansion:
            result["expanded_query"] = self._expand_query(query)

        result["keywords"] = self._extract_keywords(query)

        process_time = time.time() - start_time
        logger.info(f"Query understanding completed in {process_time:.3f}s")
        return result

    def _clean_query(self, query: str) -> str:
        """快速清理查询"""
        return re.sub(r'[^\w\s\u4e00-\u9fff]', '', query).strip()

    def _classify_intent(self, query: str) -> str:
        """快速意图分类"""
        query_lower = query.lower()
        max_score = 0
        best_intent = "课程培训提问"

        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > max_score:
                max_score = score
                best_intent = intent

        return best_intent

    def _expand_query(self, query: str) -> str:
        """轻量级查询扩展"""
        words = jieba.lcut(query)
        expanded_terms = list(words)  # 复制原词

        # 只添加最重要的同义词
        for word in words[:3]:  # 只处理前3个词
            if word in self.domain_synonyms:
                expanded_terms.extend(self.domain_synonyms[word][:1])  # 只加1个同义词

        return " ".join(expanded_terms)

    def _extract_keywords(self, query: str) -> List[str]:
        """快速关键词提取"""
        return jieba.analyse.extract_tags(query, topK=3, withWeight=False)


class OptimizedSparseRetriever:
    """优化的稀疏检索器"""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.documents = []
        self.doc_word_counts = []  # 预计算词频
        self.doc_lens = []
        self.avgdl = 0
        self.idf = {}
        self.indexed = False

    def build_index(self, documents: List[Dict[str, Any]]):
        """构建优化的索引"""
        start_time = time.time()

        self.documents = documents
        self.doc_word_counts = []
        self.doc_lens = []

        # 收集所有词汇
        all_words = set()
        doc_words_list = []

        for doc in documents:
            content = doc.get('content', '') + ' ' + doc.get('title', '')
            # 限制内容长度
            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length]

            words = jieba.lcut(content.lower())
            doc_words_list.append(words)

            word_counts = Counter(words)
            self.doc_word_counts.append(word_counts)
            self.doc_lens.append(len(words))
            all_words.update(words)

        self.avgdl = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0

        # 计算IDF
        doc_freq = {}
        for words in doc_words_list:
            unique_words = set(words)
            for word in unique_words:
                doc_freq[word] = doc_freq.get(word, 0) + 1

        N = len(documents)
        self.idf = {}
        for word in all_words:
            df = doc_freq.get(word, 0)
            self.idf[word] = math.log((N - df + 0.5) / (df + 0.5))

        self.indexed = True
        index_time = time.time() - start_time
        logger.info(f"Sparse index built in {index_time:.3f}s for {len(documents)} docs")

    def search(self, processed_query: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """优化的BM25搜索"""
        if not self.indexed:
            return []

        start_time = time.time()

        query_words = jieba.lcut(processed_query['expanded_query'].lower())
        scores = []

        k1, b = 1.5, 0.75  # BM25参数

        for i, doc in enumerate(self.documents):
            score = 0
            doc_word_counts = self.doc_word_counts[i]

            for word in query_words:
                if word in doc_word_counts and word in self.idf:
                    tf = doc_word_counts[word]
                    idf = self.idf[word]

                    # BM25公式
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * self.doc_lens[i] / self.avgdl)
                    score += idf * (numerator / denominator)

            # 标题额外加权
            title = self.documents[i].get('title', '').lower()
            title_matches = sum(1 for word in query_words if word in title)
            score += title_matches * 2.0

            if score >= self.config.sparse_min_score:  # 只保留有分数的结果
                scores.append({
                    'doc_id': i,
                    'score': score,
                    'content': self.documents[i].get('content', ''),
                    'title': self.documents[i].get('title', ''),
                    'metadata': self.documents[i].get('metadata', {})
                })

        # 按分数排序
        scores.sort(key=lambda x: x['score'], reverse=True)

        search_time = time.time() - start_time
        logger.info(f"Sparse search completed in {search_time:.3f}s, found {len(scores)} results")

        return scores[:top_k]


class OptimizedDenseRetriever:
    """优化的密集检索器 - 可选使用"""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.documents = []
        self.document_embeddings = None
        self.model = None

    def _get_model(self):
        """获取缓存的模型"""
        if self.model is None:
            self.model = get_cached_model("sentence_transformer",
                                          'paraphrase-multilingual-MiniLM-L12-v2')
        return self.model

    def build_index(self, documents: List[Dict[str, Any]]):
        """预计算文档向量"""
        if not self.config.enable_dense_retrieval:
            return

        start_time = time.time()
        model = self._get_model()
        if model is None:
            logger.warning("Dense retrieval model not available")
            return

        self.documents = documents

        # 准备文档文本
        doc_texts = []
        for doc in documents:
            text = doc.get('title', '') + ' ' + doc.get('content', '')
            # 限制长度
            if len(text) > self.config.max_content_length:
                text = text[:self.config.max_content_length]
            doc_texts.append(text)

        try:
            # 预计算所有文档的向量
            self.document_embeddings = model.encode(doc_texts, show_progress_bar=False)
            build_time = time.time() - start_time
            logger.info(f"Dense index built in {build_time:.3f}s")
        except Exception as e:
            logger.error(f"Failed to build dense index: {e}")
            self.document_embeddings = None

    def search(self, processed_query: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """快速向量搜索"""
        if not self.config.enable_dense_retrieval or self.document_embeddings is None:
            return []

        start_time = time.time()
        model = self._get_model()
        if model is None:
            return []

        try:
            # 计算查询向量
            query_embedding = model.encode([processed_query['expanded_query']])

            # 计算相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]

            # 构建结果
            results = []
            for i, sim in enumerate(similarities):
                if sim >= self.config.dense_min_score:  # 过滤低相似度结果 ，相似度设置较低得到高召回
                    results.append({
                        'doc_id': i,
                        'score': float(sim),
                        'content': self.documents[i].get('content', ''),
                        'title': self.documents[i].get('title', ''),
                        'metadata': self.documents[i].get('metadata', {})
                    })

            results.sort(key=lambda x: x['score'], reverse=True)

            search_time = time.time() - start_time
            logger.info(f"Dense search completed in {search_time:.3f}s")
            logger.info(
                f"Dense search completed in {search_time:.3f}s, found {len(results)} results above threshold {self.config.dense_min_score}")
            return results[:top_k]

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []


class FastKnowledgeRetriever:
    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        self.query_processor = QueryUnderstanding(self.config)
        self.sparse_retriever = OptimizedSparseRetriever(self.config)
        self.dense_retriever = OptimizedDenseRetriever(self.config) if self.config.enable_dense_retrieval else None
        self.is_indexed = False
        self.course_config = None
        self.documents = []

    def _is_meaningless_input(self, query: str) -> bool:
        """检查是否为无意义输入"""
        # 检查是否只包含重复字符或无意义字符
        meaningless_patterns = [
            r'^[哈嘿呵嘻嗯哦啊呀哟额嗯]+$',  # 笑声或语气词
            r'^[0-9]+$',  # 纯数字
            r'^[a-zA-Z]{1,3}$',  # 单个或少数字母
            r'^[\s\W]*$',  # 只有空格或特殊字符
            r'^(.)\1{2,}$',  # 同一字符重复3次以上
        ]

        for pattern in meaningless_patterns:
            if re.match(pattern, query):
                return True

        # 检查查询长度
        if len(query) < 2:
            return True

        return False

    async def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """主检索方法 - 优化版本：移除了延迟构建逻辑"""
        total_start_time = time.time()

        if not self.is_indexed:
            return {
                "matched": False,
                "topic": "",
                "content": "",
                "score": 0.0,
                "message": "Index not built"
            }

        # 第一层：关键词映射快速检索
        if self.config.enable_keyword_mapping:
            keyword_result = self.keyword_mapping_search(query)
            if keyword_result and keyword_result["matched"]:
                total_time = time.time() - total_start_time
                keyword_result["total_time"] = round(total_time, 4)
                logger.info(f"Keyword mapping success. Total time: {total_time:.4f}s")
                return keyword_result

        # 第二层：失败回退策略
        if self.config.fallback_strategy == "full_content":
            # 返回完整培训内容
            total_time = time.time() - total_start_time
            full_content = self.course_config.get("full_training_content", "") if self.course_config else ""
            logger.info(f"Using full content fallback in {total_time:.4f}s")
            return {
                "matched": False,
                "topic": "",
                "content": full_content,
                "score": 0.0,
                "message": "Using full training content - no keyword match found",
                "retrieval_time": round(total_time, 4)
            }

        elif self.config.fallback_strategy == "advanced_retrieval":
            # 直接使用已构建的索引进行高级检索
            return await self.advanced_retrieval(query, top_k)

        else:
            # 未知策略
            total_time = time.time() - total_start_time
            return {
                "matched": False,
                "topic": "",
                "content": "",
                "score": 0.0,
                "message": f"Unknown fallback strategy: {self.config.fallback_strategy}",
                "retrieval_time": round(total_time, 4)
            }

    def keyword_mapping_search(self, query: str) -> Optional[Dict[str, Any]]:
        """纯字符串匹配，完全不使用jieba"""
        if not self.course_config:
            return None

        topic_mapping = self.course_config.get("topic_mapping", {})
        structured_knowledge = self.course_config.get("structured_knowledge", {})

        if not topic_mapping or not structured_knowledge:
            return None

        start_time = time.time()

        # 纯字符串处理
        query_lower = query.lower().strip()
        query_clean = re.sub(r'[啊哦呢吧呀？！。，、；："""''（）【】\s]', '', query_lower)
        if self._is_meaningless_input(query_clean):
            logger.info(f"Detected meaningless input: '{query}', skipping keyword mapping")
            return None
        # 1. 完全匹配
        for key, mapped_topic in topic_mapping.items():
            key_lower = key.lower()
            key_clean = re.sub(r'[啊哦呢吧呀？！。，、；："""''（）【】\s]', '', key_lower)
            if key_clean == query_clean or key_lower == query_lower:
                if mapped_topic in structured_knowledge:
                    retrieval_time = time.time() - start_time
                    logger.info(f"Exact keyword mapping: {key} -> {mapped_topic} in {retrieval_time:.4f}s")
                    return self._create_match_result(mapped_topic, structured_knowledge[mapped_topic], 10.0,
                                                     "Exact keyword mapping", retrieval_time)

        # 2. 包含匹配
        for key, mapped_topic in topic_mapping.items():
            key_lower = key.lower()
            key_clean = re.sub(r'[啊哦呢吧呀？！。，、；："""''（）【】\s]', '', key_lower)
            if key_clean in query_clean or key_lower in query_lower or query_clean in key_clean:
                if mapped_topic in structured_knowledge:
                    retrieval_time = time.time() - start_time
                    logger.info(f"Partial keyword mapping: {key} -> {mapped_topic} in {retrieval_time:.4f}s")
                    return self._create_match_result(mapped_topic, structured_knowledge[mapped_topic], 8.0,
                                                     "Partial keyword mapping", retrieval_time)

        retrieval_time = time.time() - start_time
        logger.info(f"No keyword mapping found in {retrieval_time:.4f}s")
        return None

    def _create_match_result(self, topic: str, content: str, score: float,
                             message: str, retrieval_time: float) -> Dict[str, Any]:
        """创建匹配结果"""
        return {
            "matched": True,
            "topic": topic,
            "content": content.strip(),
            "score": score,
            "message": message,
            "retrieval_time": round(retrieval_time, 4)
        }

    async def advanced_retrieval(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """高级检索 - 使用预构建的索引"""
        if not self.is_indexed:
            return {
                "matched": False,
                "topic": "",
                "content": "",
                "score": 0.0,
                "message": "Index not built for advanced retrieval"
            }

        start_time = time.time()
        top_k = top_k or self.config.final_top_k

        # 1. 查询理解
        processed_query = await self.query_processor.process_query(query)

        # 2. 多路检索
        all_candidates = []

        if self.config.enable_sparse_retrieval:
            sparse_results = self.sparse_retriever.search(processed_query, self.config.top_k_candidates)
            for result in sparse_results:
                result['source'] = 'sparse'
            all_candidates.extend(sparse_results)

        if self.config.enable_dense_retrieval and self.dense_retriever:
            dense_results = self.dense_retriever.search(processed_query, self.config.top_k_candidates)
            for result in dense_results:
                result['source'] = 'dense'
            all_candidates.extend(dense_results)

        # 3. 结果排序和返回
        if all_candidates:
            final_results = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:top_k]
            best_result = final_results[0]
            min_score_threshold = self.config.dense_min_score if best_result.get(
                'source') == 'dense' else self.config.sparse_min_score
            total_time = time.time() - start_time
            if best_result['score'] < min_score_threshold:
                logger.info(f"Best result score {best_result['score']:.4f} below threshold {min_score_threshold}")
                return {
                    "matched": False,
                    "topic": "",
                    "content": "",
                    "score": round(best_result.get('score', 0), 4),
                    "message": f"No results above similarity threshold (score: {round(best_result['score'], 4)}, threshold: {min_score_threshold})",
                    "retrieval_time": round(total_time, 3)
                }
            logger.info(f"Advanced retrieval completed in {total_time:.3f}s")

            return {
                "matched": True,
                "topic": best_result.get('title', ''),
                "content": best_result.get('content', ''),
                "score": round(best_result.get('score', 0), 4),
                "message": f"Found via {best_result.get('source', 'unknown')} retrieval",
                "intent": processed_query.get('intent', ''),
                "entities": processed_query.get('entities', []),
                "retrieval_time": round(total_time, 3)
            }
        else:
            total_time = time.time() - start_time
            return {
                "matched": False,
                "topic": "",
                "content": "",
                "score": 0.0,
                "message": "No relevant results found in advanced retrieval",
                "retrieval_time": round(total_time, 3)
            }


# ===== 新增：新闻检索功能 =====

def extract_news_keywords(user_input: str) -> str:
    """
    智能提取新闻搜索关键词

    Args:
        user_input (str): 用户输入的原始问题

    Returns:
        str: 提取出的关键词
    """
    logger.info(f"开始提取新闻关键词，原始输入: '{user_input}'")

    # 去除常见的疑问词和语气词
    stop_words = [
        '吗', '呢', '了', '的', '在', '是', '有', '没有', '会', '能', '可以', '应该',
        '什么', '怎么', '如何', '为什么', '哪里', '哪个', '多少', '几', '？', '?', '！', '!',
        '、', '，', ',', '。', '.', '：', ':', '；', ';'
    ]

    # 去除疑问句式
    question_patterns = [
        r'.*了吗\？?$',
        r'.*呢\？?$',
        r'.*吗\？?$',
        r'^是否.*',
        r'^有没有.*',
        r'^会不会.*',
        r'^能不能.*'
    ]

    cleaned_input = user_input.strip()

    # 处理疑问句式
    for pattern in question_patterns:
        if re.match(pattern, cleaned_input):
            # 移除疑问句尾
            cleaned_input = re.sub(r'了吗\？?$|呢\？?$|吗\？?$', '', cleaned_input)
            cleaned_input = re.sub(r'^是否|^有没有|^会不会|^能不能', '', cleaned_input)
            break

    # 使用jieba进行分词和关键词提取
    words = jieba.lcut(cleaned_input)

    # 过滤停用词
    filtered_words = []
    for word in words:
        if word not in stop_words and len(word.strip()) > 0:
            filtered_words.append(word)

    # 重新组合成关键词
    if filtered_words:
        keyword = ''.join(filtered_words)
    else:
        # 如果过滤后没有词，使用原始输入去除标点符号
        keyword = re.sub(r'[？？！！。，、；：\s]+', '', user_input)

    logger.info(f"关键词提取完成: '{user_input}' -> '{keyword}'")
    logger.info(f"分词结果: {words}")
    logger.info(f"过滤后的词: {filtered_words}")

    return keyword


async def search_news_articles(keyword: str, articles_count: int = 3, days_back: int = 31) -> Dict[str, Any]:
    """
    搜索相关新闻文章

    Parameters:
        keyword (str): 搜索关键词
        articles_count (int): 返回文章数量，默认3篇
        days_back (int): 搜索最近多少天的新闻，默认31天

    Returns:
        Dict[str, Any]: 包含搜索结果的字典
    """
    logger.info(f"开始搜索新闻 - 关键词: '{keyword}', 数量: {articles_count}, 天数: {days_back}")

    try:
        # 构建请求参数
        request_params = {
            "action": "getArticles",
            "keyword": keyword,
            "sourceLocationUri": [
                "http://en.wikipedia.org/wiki/China",
                "http://en.wikipedia.org/wiki/United_States",
                "http://en.wikipedia.org/wiki/Canada",
                "http://en.wikipedia.org/wiki/United_Kingdom"
            ],
            "ignoreSourceGroupUri": "paywall/paywalled_sources",
            "articlesPage": 1,
            "articlesCount": min(articles_count, 10),  # 限制最多10篇
            "articlesSortBy": "date",
            "articlesSortByAsc": False,  # 最新的在前
            "dataType": ["news", "pr"],
            "forceMaxDataTimeWindow": days_back,
            "resultType": "articles",
            "apiKey": NEWS_API_KEY
        }

        logger.info(f"新闻API请求参数: {json.dumps(request_params, indent=2, ensure_ascii=False)}")

        # 发起异步HTTP请求
        async with aiohttp.ClientSession() as session:
            logger.info(f"发起HTTP请求到: {NEWS_API_URL}")
            async with session.post(NEWS_API_URL, json=request_params, timeout=30) as response:
                logger.info(f"HTTP响应状态码: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    logger.info(f"API返回数据结构: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

                    # 检查是否有文章结果
                    if "articles" in data and "results" in data["articles"]:
                        articles = data["articles"]["results"]
                        total_results = data["articles"].get("totalResults", 0)

                        logger.info(f"找到文章数量: {len(articles)}, 总结果数: {total_results}")

                        if not articles:
                            logger.warning(f"未找到关于'{keyword}'的相关新闻")
                            return {
                                "success": False,
                                "message": f"未找到关于'{keyword}'的相关新闻",
                                "articles": []
                            }

                        # 处理文章数据，提取关键信息
                        processed_articles = []
                        for i, article in enumerate(articles[:articles_count]):
                            logger.info(f"处理第{i + 1}篇文章: {article.get('title', '无标题')[:50]}...")

                            processed_article = {
                                "title": article.get("title", "无标题"),
                                "date": article.get("date", "未知日期"),
                                "time": article.get("time", "未知时间"),
                                "url": article.get("url", ""),
                                "source": article.get("source", {}).get("title", "未知来源"),
                                "body": article.get("body", "无内容")[:1000] + "..." if len(
                                    article.get("body", "")) > 1000 else article.get("body", "无内容"),  # 限制正文长度
                                "language": article.get("lang", "未知"),
                                "relevance": article.get("relevance", 0)
                            }
                            processed_articles.append(processed_article)

                        logger.info(f"成功处理{len(processed_articles)}篇文章")

                        return {
                            "success": True,
                            "message": f"找到 {len(processed_articles)} 篇关于'{keyword}'的相关新闻",
                            "keyword": keyword,
                            "total_results": total_results,
                            "articles": processed_articles
                        }
                    else:
                        logger.error(f"API返回数据格式异常 - 数据结构: {data}")
                        return {
                            "success": False,
                            "message": f"API返回数据格式异常，未找到关于'{keyword}'的新闻",
                            "articles": []
                        }
                else:
                    logger.error(f"新闻API请求失败 - 状态码: {response.status}, 响应内容: {await response.text()}")
                    return {
                        "success": False,
                        "message": f"新闻API请求失败，状态码: {response.status}",
                        "articles": []
                    }

    except asyncio.TimeoutError:
        logger.error("新闻API请求超时")
        return {
            "success": False,
            "message": "新闻API请求超时，请稍后重试",
            "articles": []
        }
    except Exception as e:
        logger.error(f"搜索新闻时发生错误: {str(e)}", exc_info=True)
        return {
            "success": False,
            "message": f"搜索新闻时发生错误: {str(e)}",
            "articles": []
        }


def summarize_news_articles(news_result: Dict[str, Any]) -> str:
    """
    总结新闻文章内容

    Parameters:
        news_result (Dict[str, Any]): 新闻搜索结果

    Returns:
        str: 新闻总结
    """
    logger.info("开始生成新闻总结")

    if not news_result.get("success") or not news_result.get("articles"):
        error_message = f"新闻搜索失败: {news_result.get('message', '未知错误')}"
        logger.info(f"新闻总结结果: {error_message}")
        return error_message

    articles = news_result["articles"]
    keyword = news_result.get("keyword", "相关主题")

    summary_parts = [
        f"🔍 关于'{keyword}'的最新新闻总结:",
        f"📊 共找到 {news_result.get('total_results', 0)} 篇相关报道，以下是最新的 {len(articles)} 篇:",
        ""
    ]

    for i, article in enumerate(articles, 1):
        summary_parts.extend([
            f"📰 新闻 {i}:",
            f"   标题: {article['title']}",
            f"   时间: {article['date']} {article['time']}",
            f"   来源: {article['source']}",
            f"   语言: {article['language']}",
            f"   摘要: {article['body'][:200]}{'...' if len(article['body']) > 200 else ''}",
            f"   链接: {article['url']}",
            ""
        ])

    summary_parts.append("💡 这些是您询问的新闻内容吗？如果需要了解具体某篇新闻的详细内容，请告诉我。")

    summary = "\n".join(summary_parts)
    logger.info(f"新闻总结生成完成，总长度: {len(summary)} 字符")

    return summary


# 新增：新闻搜索函数（function call入口）
async def get_latest_news(keyword: str, count: int = 3, days: int = 31) -> Dict[str, Any]:
    """搜索最新相关新闻

    Parameters:
        keyword (str): 搜索关键词
        count (int): 返回新闻数量，默认3篇，最多10篇
        days (int): 搜索最近多少天的新闻，默认31天

    Returns:
        Dict[str, Any]: 包含新闻搜索结果和总结的字典
    """
    logger.info(f"=== 开始新闻搜索函数调用 ===")
    logger.info(f"入参 - 关键词: '{keyword}', 数量: {count}, 天数: {days}")

    try:
        # 限制参数范围
        count = min(max(count, 1), 10)  # 1-10篇
        days = min(max(days, 1), 90)  # 1-90天

        logger.info(f"参数范围限制后 - 数量: {count}, 天数: {days}")

        # 搜索新闻
        logger.info("开始调用 search_news_articles")
        news_result = await search_news_articles(keyword, count, days)
        logger.info(
            f"search_news_articles 返回结果: success={news_result.get('success')}, 文章数={len(news_result.get('articles', []))}")

        # 生成总结
        logger.info("开始生成新闻总结")
        summary = summarize_news_articles(news_result)
        logger.info("新闻总结生成完成")

        final_result = {
            "success": news_result["success"],
            "keyword": keyword,
            "search_params": {
                "count": count,
                "days": days
            },
            "raw_result": news_result,
            "summary": summary,
            "message": news_result.get("message", "")
        }

        logger.info(f"=== 新闻搜索函数调用完成 ===")
        logger.info(f"最终结果: success={final_result['success']}, 关键词='{final_result['keyword']}'")

        return final_result

    except Exception as e:
        logger.error(f"新闻搜索函数发生异常: {str(e)}", exc_info=True)
        error_result = {
            "success": False,
            "keyword": keyword,
            "error": str(e),
            "summary": f"搜索关于'{keyword}'的新闻时发生错误: {str(e)}",
            "message": "新闻搜索功能暂时不可用"
        }

        logger.info(f"=== 新闻搜索函数调用异常结束 ===")
        return error_result


# 创建全局服务实例（启动时构建索引）
_GLOBAL_SERVICE_INSTANCE = None


def get_service_instance():
    """获取全局服务实例"""
    global _GLOBAL_SERVICE_INSTANCE
    if _GLOBAL_SERVICE_INSTANCE is None:
        logger.info("Creating global service instance...")
        _GLOBAL_SERVICE_INSTANCE = OptimizedRealtimeFunctionCallService()
    return _GLOBAL_SERVICE_INSTANCE


# 原有接口适配
class FunctionCallQuery(BaseModel):
    user_input: str
    function_call_name: str


class RealtimeFunctionCallInfo(BaseModel):
    topic: str
    action: str
    query: Optional[FunctionCallQuery] = None


class OptimizedRealtimeFunctionCallService:
    def __init__(self):
        self.prompt_config = self.load_prompt_config()
        self.courses_config = self.prompt_config.get("azure_realtime_function_call", {})
        self.retrievers = {}  # 缓存检索器
        # 新增：启动时预构建所有课程的索引
        self._prebuild_all_indexes()

    def _prebuild_all_indexes(self):
        """预构建所有课程的索引"""
        logger.info("Starting to prebuild indexes for all courses...")
        start_time = time.time()

        for topic, course_config in self.courses_config.items():
            try:
                # 为每个主题预构建两种策略的索引
                for strategy in ["full_content", "advanced_retrieval"]:
                    cache_key = f"{topic}_{strategy}"
                    if cache_key not in self.retrievers:
                        logger.info(f"Prebuilding index for {topic} with {strategy} strategy...")
                        retriever = self._create_and_build_retriever(topic, course_config, strategy)
                        self.retrievers[cache_key] = retriever

            except Exception as e:
                logger.error(f"Failed to prebuild index for {topic}: {e}")

        total_time = time.time() - start_time
        logger.info(f"Finished prebuilding indexes in {total_time:.2f}s")

    def _create_and_build_retriever(self, topic: str, course_config: Dict[str, Any],
                                    fallback_strategy: str) -> FastKnowledgeRetriever:
        """创建并构建检索器（包括索引）"""
        if fallback_strategy == "full_content":
            # 轻量级配置，只需要关键词映射
            config = RetrievalConfig(
                enable_keyword_mapping=True,
                enable_query_understanding=False,
                enable_sparse_retrieval=False,
                enable_dense_retrieval=False,
                enable_reranking=False,
                fallback_strategy=fallback_strategy,
                enable_caching=True
            )

            retriever = FastKnowledgeRetriever(config)
            retriever.course_config = course_config
            retriever.documents = []
            retriever.is_indexed = True

        else:  # advanced_retrieval
            # 完整配置
            config = RetrievalConfig(
                enable_keyword_mapping=True,
                enable_query_understanding=True,
                enable_sparse_retrieval=True,
                enable_dense_retrieval=True,
                enable_reranking=False,  # 可选
                fallback_strategy=fallback_strategy,
                top_k_candidates=10,
                final_top_k=5,
                max_content_length=800,
                enable_caching=True
            )

            retriever = FastKnowledgeRetriever(config)
            retriever.course_config = course_config

            # 立即构建文档和索引
            documents = []
            structured_knowledge = course_config.get("structured_knowledge", {})
            for title, content in structured_knowledge.items():
                documents.append({
                    'title': title,
                    'content': content,
                    'metadata': {'topic': topic}
                })

            if documents:
                # 构建所有索引
                retriever.documents = documents
                retriever.is_indexed = True

                # 构建稀疏索引
                if config.enable_sparse_retrieval:
                    retriever.sparse_retriever.build_index(documents)

                # 构建密集索引
                if config.enable_dense_retrieval:
                    retriever.dense_retriever.build_index(documents)

        return retriever

    def get_or_create_retriever(self, topic: str, course_config: Dict[str, Any],
                                fallback_strategy: str = "full_content") -> FastKnowledgeRetriever:
        """获取或创建检索器 - 优化版本：使用预构建的索引"""
        cache_key = f"{topic}_{fallback_strategy}"

        # 如果已经有预构建的检索器，直接返回
        if cache_key in self.retrievers:
            logger.info(f"Using pre-built retriever for {topic} with {fallback_strategy}")
            return self.retrievers[cache_key]

        # 如果没有预构建（例如新增的主题），则创建并构建
        logger.warning(f"No pre-built retriever found for {topic}, building now...")
        retriever = self._create_and_build_retriever(topic, course_config, fallback_strategy)
        self.retrievers[cache_key] = retriever

        return retriever

    @staticmethod
    def load_prompt_config():
        try:
            with open('./utils/prompt.json', encoding='utf-8') as config_file:
                return json.load(config_file)
        except Exception as e:
            logger.error(f"Failed to load prompt config file: {e}")
            return {}

    def get_course_config(self, topic: str) -> Optional[Dict[str, Any]]:
        if topic in self.courses_config:
            return self.courses_config[topic]
        return None

    def start_session(self, topic: str) -> Dict[str, Any]:
        course_config = self.get_course_config(topic)
        if course_config:
            return {
                "instructions": course_config.get("instructions", ""),
                "tools": course_config.get("tools", [])
            }
        return {}

    async def get_function_call_result(self, query: FunctionCallQuery, topic: str,
                                       fallback_strategy: str = "full_content") -> Dict[str, Any]:
        """获取函数调用结果 - 优化返回格式"""
        logger.info(f"=== 开始处理函数调用 ===")
        logger.info(f"主题: {topic}")
        logger.info(f"回退策略: {fallback_strategy}")
        logger.info(f"函数名: {query.function_call_name}")
        logger.info(f"用户输入: '{query.user_input}'")

        course_config = self.get_course_config(topic)
        if not course_config:
            logger.error(f"Course config not found for topic '{topic}'")
            return {"result": "", "error": "Topic not found"}

        func_name = query.function_call_name
        user_input = query.user_input

        logger.info(f"Processing function call: {func_name} with input: {user_input}, fallback: {fallback_strategy}")

        # 处理新闻搜索函数调用
        if func_name == "get_latest_news":
            logger.info("识别为新闻搜索函数调用")
            try:
                # 智能提取关键词
                logger.info("开始提取搜索关键词")
                keyword = extract_news_keywords(user_input)
                logger.info(f"提取的关键词: '{keyword}'")

                # 调用新闻搜索函数
                logger.info("开始调用新闻搜索API")
                news_result = await get_latest_news(keyword, count=3, days=31)
                logger.info(f"新闻搜索完成: success={news_result.get('success')}")

                # 构建统一格式的返回结果
                if news_result.get("success"):
                    result_content = news_result.get("summary", "")
                else:
                    result_content = ""

                final_result = {
                    "result": result_content,
                    func_name: news_result
                }
                logger.info(f"=== 新闻搜索函数调用完成 ===")
                return final_result

            except Exception as e:
                logger.error(f"新闻搜索函数调用异常: {str(e)}", exc_info=True)
                error_result = {
                    "result": "",
                    func_name: {
                        "success": False,
                        "keyword": user_input,
                        "error": str(e),
                        "summary": f"搜索新闻时发生错误: {str(e)}",
                        "message": "新闻搜索功能暂时不可用"
                    }
                }
                logger.info(f"=== 新闻搜索函数调用异常结束 ===")
                return error_result

        # 处理培训知识检索函数调用
        elif func_name in ["get_enterprise_going_global_info", "get_huiren_training_info"]:
            logger.info(f"识别为培训知识检索函数调用: {func_name}")

            # 获取预构建的检索器
            logger.info("获取预构建的检索器")
            retriever = self.get_or_create_retriever(topic, course_config, fallback_strategy)
            if not retriever:
                logger.error("检索器获取失败")
                return {"result": "", "error": "Retriever not available"}

            # 执行检索
            logger.info("开始执行知识检索")
            knowledge_result = await retriever.retrieve(user_input)
            logger.info(
                f"知识检索完成: matched={knowledge_result.get('matched')}, topic='{knowledge_result.get('topic')}'")

            # 构建统一格式的返回结果
            if knowledge_result.get("matched") or knowledge_result.get("content"):
                result_content = knowledge_result.get("content", "")
            else:
                result_content = ""

            final_result = {
                "result": result_content,
                func_name: knowledge_result
            }
            logger.info(f"=== 培训知识检索函数调用完成 ===")
            return final_result

        else:
            logger.error(f"Unknown function call: {func_name}")
            logger.info(f"=== 未知函数调用结束 ===")
            return {"result": "", "error": f"Unknown function: {func_name}"}


async def realtime_function_call(fc_info: RealtimeFunctionCallInfo,
                                 fallback_strategy: str = "full_content") -> Dict[str, Any]:
    """处理实时函数调用接口 - 优化版本：使用全局服务实例，返回统一格式"""
    logger.info("==================== 函数调用开始 ====================")
    logger.info(f"请求信息:")
    logger.info(f"  - 主题 (topic): {fc_info.topic}")
    logger.info(f"  - 动作 (action): {fc_info.action}")
    logger.info(f"  - 回退策略 (fallback_strategy): {fallback_strategy}")

    if fc_info.query:
        logger.info(f"  - 查询信息:")
        logger.info(f"    - 用户输入: '{fc_info.query.user_input}'")
        logger.info(f"    - 函数名称: {fc_info.query.function_call_name}")

    try:
        # 使用全局服务实例（已预构建索引）
        service = get_service_instance()

        if fc_info.action == "startSession":
            logger.info("执行启动会话操作")
            result = service.start_session(fc_info.topic)
            logger.info("Start session success")
            logger.info("==================== 函数调用结束 ====================")
            return result

        elif fc_info.action == "getFunctionCallResult":
            if not fc_info.query:
                logger.error("获取函数调用结果时缺少query参数")
                logger.info("==================== 函数调用异常结束 ====================")
                return {"result": "", "error": "Missing query parameter"}

            logger.info("开始获取函数调用结果")
            result = await service.get_function_call_result(fc_info.query, fc_info.topic, fallback_strategy)
            logger.info("Get function call result success")
            logger.info(f"返回结果包含result字段: {bool(result.get('result'))}")
            logger.info("==================== 函数调用结束 ====================")
            return result

        else:
            logger.error(f"未定义的动作: {fc_info.action}")
            logger.info("==================== 函数调用异常结束 ====================")
            return {"result": "", "error": f"Unknown action: {fc_info.action}"}

    except Exception as e:
        logger.error(f"处理实时函数调用时发生错误: {str(e)}", exc_info=True)
        logger.info("==================== 函数调用异常结束 ====================")
        return {"result": "", "error": f"Internal error: {str(e)}"}
