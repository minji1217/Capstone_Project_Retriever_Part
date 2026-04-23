import numpy as np
import time 
import config
import utils
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer
from fusion import rank_fusion

class OnlinePaperProcess:
    def __init__(self):
        # 1. 모든 모듈 로드
        print("[모듈 로드 중 ...]")
        start = time.time()
        self.query_builder = QueryBuilder()
        self.embedder = SpecterEmbedder()
        self.retriever = FaissRetriever()
        self.bib_scorer = SoftBiasScorer()

        print(f"[로딩 완료 ({time.time() - start: .2f}초 소요)]")

    def get_features(self, user_input):
        '''
        프론트엔드에서 온 데이터 받아서 최종 피처 반환
        - user_input : { "title" : str, "abstract" : str, "context" : str, "bib_ids" : list}
        '''
        req_id = "req_" + utils.get_timestamp()

        # 1. 쿼리 생성 
        p_query, c_query = self.query_builder.build_online_query(
            user_input_text=user_input.get('context', ''),
            title=user_input.get('title', ''),
            abstract=user_input.get('abstract', '')
        )

        # 2. 임베딩 연산
        vecs = self.embedder.encode([p_query, c_query])
        p_vec, c_vec = vecs[0:1], vecs[1:2]

        # 3. FAISS 검색
        p_res = self.retriever.search(p_vec, [req_id], source = ["paper"])
        c_res = self.retriever.search(c_vec, [req_id], source = ["context"])

        # 4. RRF 융합
        fused = rank_fusion(p_res, c_res)[0]

        # 5. Soft bias 계산
        bib_ids = user_input.get('bib_ids', [])
        biased = self.bib_scorer.soft_bias(fused, bib_ids)

        # 6. 피처 정규화 
        raw_sims = [c['sim'] for c in biased]
        raw_bibs = [c.get('bib_score', 0.0) for c in biased]

        norm_sims = utils.normalize(raw_sims)
        norm_bibs = utils.normalize(np.log1p(raw_bibs))

        # 7. 최종 데이터 형식 가공
        clean_candidates = []
        for idx, cand in enumerate(biased):
            clean_candidates.append({
                "paper_id" : str(cand['paper_id']),
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            })

        return {
            "query_id": req_id,
            "candidates": clean_candidates
        }
    
if __name__ == "__main__":
    # 서버 객체 생성
    engine = OnlinePaperProcess()
    
    # 프론트엔드에서 날아온 가상의 JSON 데이터
    sample = {
        "title": "Large Language Models for RecSys",
        "abstract": "This paper explores...",
        "context_chunk": "기존의 NCF 모델의 한계를 극복하기 위해 최근 LLM을 활용한 추천 시스템이 각광받고 있다 \\cite{",
        "bib_ids": ["Koren2009", "He2017"]
    }
    
    print("\n 실시간 피처 추출 시작...")
    start_time = time.time()
    
    # 파이프라인 시작
    result = engine.get_features(sample)
    
    print(f"처리 완료. 소요 시간: {time.time() - start_time:.4f}초")
    print(f"반환된 후보 개수: {len(result['candidates'])}개")
    print(f"첫 번째 후보 샘플: {result['candidates'][0]}")


    