import json
import re
import math
import asyncio
import time
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
from utils.nlp_logging import CustomLogger
import jieba
import jieba.analyse
from collections import Counter

logger = CustomLogger(name="DaoTest optimized retrieval api", write_to_file=True)

# 全局模型缓存 - 避免重复加载
_GLOBAL_MODEL_CACHE = {}


def get_cached_model(model_type: str, model_name: str = None):
    """获取缓存的模型"""
    cache_key = f"{model_type}_{model_name}" if model_name else model_type

    if cache_key not in _GLOBAL_MODEL_CACHE:
        try:
            if model_type == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name or 'paraphrase-multilingual-MiniLM-L12-v2')
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
    enable_query_understanding: bool = True
    enable_query_expansion: bool = True
    enable_intent_classification: bool = True
    enable_entity_extraction: bool = True

    enable_sparse_retrieval: bool = True
    enable_dense_retrieval: bool = False  # 默认关闭，节省时间

    enable_reranking: bool = False  # 默认关闭，节省时间
    rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    sparse_weight: float = 1.0  # 只用稀疏检索时给满权重
    dense_weight: float = 0.0
    rerank_weight: float = 0.0

    top_k_candidates: int = 10
    final_top_k: int = 5

    # 性能优化配置
    enable_async: bool = True
    max_content_length: int = 1000  # 限制内容长度
    enable_caching: bool = True


class QueryUnderstanding:
    """查询理解层 - 轻量化版本"""

    def __init__(self, config: RetrievalConfig):
        self.config = config
        # 简化的意图关键词
        self.intent_keywords = {
            "风险评估": ["风险", "危险", "问题", "困难", "挑战", "不利", "影响", "后果"],
            "政策咨询": ["政策", "法规", "规定", "要求", "条件"],
            "市场分析": ["市场", "机会", "前景", "趋势", "竞争"],
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
            "intent": "一般咨询",
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
        best_intent = "一般咨询"

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

            if score > 0:  # 只保留有分数的结果
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
                if sim > 0.1:  # 过滤低相似度结果
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

            return results[:top_k]

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []


class FastKnowledgeRetriever:
    """快速知识检索器"""

    def __init__(self, config: RetrievalConfig = None):
        self.config = config or RetrievalConfig()
        self.query_processor = QueryUnderstanding(self.config)
        self.sparse_retriever = OptimizedSparseRetriever(self.config)
        self.dense_retriever = OptimizedDenseRetriever(self.config) if self.config.enable_dense_retrieval else None
        self.is_indexed = False

    def build_index(self, documents: List[Dict[str, Any]]):
        """快速构建索引"""
        start_time = time.time()
        logger.info(f"Building fast index for {len(documents)} documents")

        if self.config.enable_sparse_retrieval:
            self.sparse_retriever.build_index(documents)

        if self.config.enable_dense_retrieval and self.dense_retriever:
            self.dense_retriever.build_index(documents)

        self.is_indexed = True
        total_time = time.time() - start_time
        logger.info(f"Fast index building completed in {total_time:.3f}s")

    async def retrieve(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """快速检索"""
        total_start_time = time.time()

        if not self.is_indexed:
            return {
                "matched": False,
                "topic": "",
                "content": "",
                "score": 0.0,
                "message": "Index not built"
            }

        top_k = top_k or self.config.final_top_k

        # 1. 查询理解
        processed_query = await self.query_processor.process_query(query)

        # 2. 检索（优先使用稀疏检索，速度快）
        all_candidates = []

        if self.config.enable_sparse_retrieval:
            sparse_results = self.sparse_retriever.search(processed_query, self.config.top_k_candidates)
            for result in sparse_results:
                result['source'] = 'sparse'
            all_candidates.extend(sparse_results)

        # 只有在稀疏检索结果不足时才使用密集检索
        if self.config.enable_dense_retrieval and len(all_candidates) < 3 and self.dense_retriever:
            dense_results = self.dense_retriever.search(processed_query, self.config.top_k_candidates)
            for result in dense_results:
                result['source'] = 'dense'
            all_candidates.extend(dense_results)

        # 3. 简单排序（跳过重排序以节省时间）
        if all_candidates:
            # 按分数排序
            final_results = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:top_k]

            best_result = final_results[0]
            total_time = time.time() - total_start_time

            logger.info(f"Fast retrieval completed in {total_time:.3f}s")

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
            total_time = time.time() - total_start_time
            return {
                "matched": False,
                "topic": "",
                "content": "",
                "score": 0.0,
                "message": "No relevant results found",
                "retrieval_time": round(total_time, 3)
            }


# 原有接口适配 - 使用快速版本
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

    def get_or_create_retriever(self, topic: str, course_config: Dict[str, Any]) -> FastKnowledgeRetriever:
        """获取或创建快速检索器"""
        if topic not in self.retrievers:
            # 快速配置 - 优先速度
            config = RetrievalConfig(
                enable_query_understanding=True,
                enable_sparse_retrieval=True,
                enable_dense_retrieval=True,  # 关闭密集检索
                enable_reranking=True,  # 关闭重排序
                top_k_candidates=5,
                final_top_k=1,
                max_content_length=800,  # 限制内容长度
                enable_caching=True
            )

            retriever = FastKnowledgeRetriever(config)

            # 构建文档索引
            documents = []
            structured_knowledge = course_config.get("structured_knowledge", {})

            for title, content in structured_knowledge.items():
                documents.append({
                    'title': title,
                    'content': content,
                    'metadata': {'topic': topic}
                })

            if documents:
                retriever.build_index(documents)
                self.retrievers[topic] = retriever
                logger.info(f"Created fast retriever for topic: {topic}")
            else:
                logger.warning(f"No documents found for topic: {topic}")
                return None

        return self.retrievers.get(topic)

    def start_session(self, topic: str) -> Dict[str, Any]:
        course_config = self.get_course_config(topic)
        if course_config:
            return {
                "instructions": course_config.get("instructions", ""),
                "tools": course_config.get("tools", [])
            }
        return {}

    async def get_function_call_result(self, query: FunctionCallQuery, topic: str) -> Dict[str, Any]:
        course_config = self.get_course_config(topic)
        if not course_config:
            logger.error(f"Course config not found for topic '{topic}'")
            return {}

        # 获取快速检索器
        retriever = self.get_or_create_retriever(topic, course_config)
        if not retriever:
            return {}

        func_name = query.function_call_name
        user_input = query.user_input

        logger.info(f"Processing fast function call: {func_name} with user input: {user_input}")

        # 执行快速检索
        result = await retriever.retrieve(user_input)

        # 如果没有匹配到，返回完整培训内容
        if not result["matched"]:
            full_content = course_config.get("full_training_content", "")
            result["content"] = full_content

        return {func_name: result}


async def realtime_function_call(fc_info: RealtimeFunctionCallInfo) -> Dict[str, Any]:
    """处理实时函数调用接口 - 优化版"""
    logger.info("------------------start--------------------")
    logger.info(f"Received fast retrieval request: topic={fc_info.topic}, action={fc_info.action}")

    try:
        service = OptimizedRealtimeFunctionCallService()

        if fc_info.action == "startSession":
            result = service.start_session(fc_info.topic)
            logger.info("Start session success")
            return result

        elif fc_info.action == "getFunctionCallResult":
            if not fc_info.query:
                logger.error("Must provide query parameter when getting function call result")
                return {}

            result = await service.get_function_call_result(fc_info.query, fc_info.topic)
            logger.info("Get function call result success")
            return result

        else:
            logger.error(f"Action error, undefined action: {fc_info.action}")
            return {}

    except Exception as e:
        logger.error(f"Error occurred while processing realtime function call: {str(e)}")
        return {}