# ====== 知识库引擎（RAG） ======
import json, re, math
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from config import DOC_DIR, INDEX_DIR


class KnowledgeBase:
    """基于 TF-IDF 的本地知识库，支持文档增删和搜索。"""

    def __init__(self):
        self.chunks = []
        self.chunk_sources = []
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words=None,
                                          analyzer="char_wb", ngram_range=(1, 3))
        self._dirty = True

    def add_document(self, filepath: str) -> int:
        """添加文档到知识库，返回切分后的片段数。"""
        path = Path(filepath)
        text = path.read_text(encoding="utf-8")
        chunks = self._chunk_text(text)
        name = path.stem
        self.chunks.extend(chunks)
        self.chunk_sources.extend([name] * len(chunks))
        self._dirty = True
        return len(chunks)

    def remove_document(self, name: str):
        """从知识库删除指定名称的文档。"""
        new_chunks = []
        new_sources = []
        for c, s in zip(self.chunks, self.chunk_sources):
            if s != name:
                new_chunks.append(c)
                new_sources.append(s)
        self.chunks = new_chunks
        self.chunk_sources = new_sources
        self._dirty = True

    def _chunk_text(self, text: str, size=500, overlap=100) -> list:
        """将文本切分成重叠的片段。"""
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start += size - overlap
        return chunks

    def search(self, query: str, top_k=3) -> list:
        """搜索知识库，返回最相关的片段。"""
        if not self.chunks:
            return []
        if self._dirty:
            self._rebuild_index()
        q_vec = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix * q_vec.T).toarray().flatten()
        top = scores.argsort()[::-1][:top_k]
        results = []
        for i in top:
            if scores[i] > 0:
                results.append({
                    "source": self.chunk_sources[i],
                    "score": round(scores[i], 4),
                    "content": self.chunks[i][:300]
                })
        return results

    def _rebuild_index(self):
        """重建 TF-IDF 索引。"""
        if self.chunks:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        else:
            self.tfidf_matrix = None
        self._dirty = False

    def get_document_list(self) -> list:
        """获取文档列表。"""
        seen = set()
        result = []
        for name in self.chunk_sources:
            if name not in seen:
                seen.add(name)
                fpath = DOC_DIR / (name + ".txt")
                size = fpath.stat().st_size if fpath.exists() else 0
                mtime = (datetime.fromtimestamp(fpath.stat().st_mtime)
                         .strftime("%m-%d %H:%M") if fpath.exists() else "")
                result.append({"name": name, "size": size, "updated": mtime})
        return result


# 全局知识库实例
kb = KnowledgeBase()

# 启动时加载已有文档
for f in DOC_DIR.glob("*.txt"):
    kb.add_document(str(f))
