import numpy as np
import config 
import time 
import utils
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer

def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer):
    # 1. 단일 쿼리용 리스트 준비
    unified_query_list = [] # context + title + abstract
    metadata_list = []

    for item in paper_batch:
        paper_id = item.get('paper_id', '')
        title = item.get('title', '')
        abstract = item.get('abstract', '')

        # QueryBuilder 통해 해당 논문의 모든 인용구 추출
        _, context_queries = query_builder.extract_all_samples(
            paper_id, item.get('full_text', ''), title, abstract, item.get('all_references', [])
        )

        for sample in context_queries:
            safe_context = utils.truncate_words(sample['context_query'], 150)
            safe_title = utils.truncate_words(title, 30)
            safe_abstract = utils.truncate_words(abstract, 200)

            # context first 결합 : context [SEP] title [SEP] abstract
            unified_text = f"{safe_context} [SEP] {safe_title} [SEP] {safe_abstract}"

            unified_query_list.append(unified_text)
            metadata_list.append(sample)

    total_samples = len(unified_query_list)
    print(f"추출된 context 기반 unified 쿼리 개수 : {total_samples}개")

    # 인용구가 하나도 없다면 패스 
    if total_samples == 0: return []

    # 2. 단일 배치 임베딩
    u_vectors = embedder.encode(unified_query_list)

    # 3. 단일 FAISS 검색
    query_ids = [m['query_id'] for m in metadata_list]
    search_results = retriever.search(u_vectors, query_ids, source = ["unified"] * total_samples, top_k = 100)

    final_output_for_next = []

    # 4. Soft bias 적용
    for i in range(total_samples):
        meta = metadata_list[i]

        # FAISS에서 뽑힌 1개의 top-100 결과 
        candidates = search_results[i]

        # soft bias 계산 (bib_ids 사용)
        user_bibs = meta.get('bib_ids', [])
        biased = bib_scorer.soft_bias(candidates, user_bibs)

        # 정규화 
        raw_sims = [c['sim'] for c in biased]
        raw_bibs = [c.get('bib_score', 0.0) for c in biased]

        norm_sims = utils.normalize(raw_sims)
        norm_bibs = utils.normalize(np.log1p(raw_bibs))

        # top-100 각 논문에 대해 필요한 피처만 추출
        clean_candidates = []
        for idx, cand in enumerate(biased):
            clean_candidates.append({
                "paper_id": str(cand['paper_id']),
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            })

        query_packet = {
            "query_id": meta['query_id'],
            'target_ids': meta['target_ids'],
            'candidates': clean_candidates
        }
        
        final_output_for_next.append(query_packet)

    return final_output_for_next

def run_pipeline(data_path, paper_batch_size):
    '''
    [동작 방식] 전체 데이터셋을 논문 단위로 쪼개고, 논문 내에서도 context 단위로 쪼개어 통합(Unified) 검색 동작
    '''
    print(f"[Offline Unified 실험용 추천 파이프라인 가동 시작...] (데이터: {data_path})")
    start_time = time.time()

    # 1. 모듈 생성 
    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    # 2. 데이터셋 로드
    eval_data = utils.load_json(data_path)
    total_papers = len(eval_data)
    all_processed_queries = [] 

    print(f"총 논문 개수 : {total_papers}개 (논문 {paper_batch_size}개씩 묶어서 처리)")

    # 3. 파이프라인 실행
    for i in range(0, total_papers, paper_batch_size):
        paper_batch = eval_data[i : i + paper_batch_size]
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer)
        all_processed_queries.extend(batch_results)
    
    print(f"[최종 성능 평가 결과]")
    print(f"총 소요시간 : {time.time() - start_time: .2f}초")

    return all_processed_queries

if __name__ == "__main__":
    final_data = run_pipeline(config.EVAL_DATA_PATH, config.PAPER_BATCH_SIZE)
    utils.save_json(final_data, "offline_unified_output.json") 
    print("'offline_unified_output.json' 저장 완료")