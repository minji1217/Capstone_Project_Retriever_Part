import numpy as np
import config 
import time 
import utils
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from fusion import rank_fusion
from soft_bias import SoftBiasScorer


def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer):
    # paper_batch : eval_data에서 32개 논문 가져온 리스트 (json 형태)
    # 1. Flatten : 논문 32개 각각의 모든 context를 1차원 리스트로 모음
    unique_paper_queries = {} # paper query 저장 (최대 32개)
    context_query_list = []   # context query 저장 
    metadata_list = []        # 메타데이터 보관소 (QueryBuilder가 만든 딕셔너리 결과물)

    for item in paper_batch:
        paper_id = item.get('paper_id', '')

        # QueryBuilder를 통해 paper query 1개, context query N개 추출 
        paper_query, context_queries = query_builder.extract_all_samples(
            paper_id, item.get('full_text',''), item.get('title', ''), item.get('abstract',''), item.get('all_references', [])
        )
        

        # 전역 쿼리는 논문당 1번만 저장
        unique_paper_queries[paper_id] = paper_query

        # 해당 논문의 모든 인용구([CITE:])를 context_query_list에 저장
        for sample in context_queries:
            context_query_list.append(sample['context_query'])
            metadata_list.append(sample) 

    total_samples = len(context_query_list)
    print(f"context 개수: {len(context_query_list)}")

    # 가져온 논문 32개 모두 인용구 하나도 없다면 패스 
    if total_samples == 0: return []

    # 2. 배치 임베딩
    # 2-1. context query 한 번에 임베딩 
    # 이때, embedder 내부에서 batch_size(예: 64) 단위로 쪼개어 연산 후 붙여줌
    c_vectors = embedder.encode(context_query_list)

    # 2-2. 전역 쿼리는 중복 제거한 paper_batch(예: 32) 개수만큼 인코딩
    # 기존 : {"paper_id : paper_query", " : ", ...}
    # 적용 후 : [[, , ,], [, , ,], ...]
    p_unique_vecs = embedder.encode(list(unique_paper_queries.values()))

    # 2-3. 인코딩된 paper query 벡터들(32개)을 딕셔너리로 변형
    p_vec_dict = {pid: vec for pid, vec in zip(unique_paper_queries.keys(), p_unique_vecs)}

    # 2-4. paper query 벡터를 context query 개수에 맞춰 복제 
    p_vectors = np.array([p_vec_dict[m['paper_id']] for m in metadata_list])

    # 3. FAISS 검색 
    query_ids = [m['query_id'] for m in metadata_list]

    # 3-1. FAISS가 각 query(paper/context)의 top-100 결과 도출
    p_search_results = retriever.search(p_vectors, query_ids, source = ["paper"] * total_samples)
    c_search_results = retriever.search(c_vectors, query_ids, source = ["context"] * total_samples)

    # 4. RRF, Soft Bias 적용 및 채점 
    batch_metrics = [] # 해당 배치에서 나온 각 쿼리에서의 metrics 점수 저장

    # 4-1. RRF
    # input : [[{}, {}, ...], [{}, {}, ...], ...] 2개 
    # output : "" 1개 
    all_fused_results = rank_fusion(p_search_results, c_search_results)
    final_output_for_next = [] # 다음 단계에 제공 

    # 4-2. Soft Bias 
    for i in range(total_samples):
        meta = metadata_list[i]

        fused = all_fused_results[i] # fusion 적용하고 난 top-100

        # soft bias 계산 (bib_ids 사용)
        user_bibs = meta.get('bib_ids', [])
        biased = bib_scorer.soft_bias(fused, user_bibs)

        # sim, bib_score 정규화 
        raw_sims = [c['sim'] for c in biased]
        raw_bibs = [c.get('bib_score', 0.0) for c in biased]

        norm_sims = utils.normalize(raw_sims)
        norm_bibs = utils.normalize(np.log1p(raw_bibs))

        # top-100 각 논문에 대해 필요한 피처만 추출
        clean_candidates = []
        for idx, cand in enumerate(biased):
            clean_candidates.append({
                "paper_id": cand['paper_id'],
                # "rrf_score": cand['rrf_score'],
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
    [동작 방식] 전체 데이터셋을 논문 단위로 쪼개고, 논문 내에서도 context 단위로 쪼개어 동작
    '''
    print("f[Offline 실험용 추천 파이프라인 가동 시작...] (데이터: {data_path}")
    start_time = time.time()

    # 1. 모듈 생성 
    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    # 2. 데이터셋 로드 (정답지 포함된 JSON 파일)
    eval_data = utils.load_json(data_path)
    total_papers = len(eval_data)
    all_processed_queries = [] # 모든 배치를 1차원으로 통합할 리스트 (할지말지 고민)

    print(f"총 논문 개수 : {total_papers}개 (논문 {paper_batch_size}개씩 묶어서 처리")

    # 3. 데이터셋 순회하며 파이프라인 실행 (paper_batch_size(예: 32) 단위로 쪼갬)
    for i in range(0, total_papers, paper_batch_size):
        paper_batch = eval_data[i : i + paper_batch_size]
        # 논문 100개, batch : 32일때 마지막 루프 i=96일땐 96~128(96+32)가 아닌 96~100이어야하므로 min 취함 
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer)
        
        all_processed_queries.extend(batch_results)# 다음 파트에 합치기 (batch_results 이용할지 말지)
    
    
    print(f"[최종 성능 평가 결과]")
    print(f"총 소요시간 : {time.time() - start_time: .2f}초")

    return all_processed_queries

    

if __name__ == "__main__":
    final_data = run_pipeline(config.EVAL_DATA_PATH, config.PAPER_BATCH_SIZE)
    utils.save_json(final_data, "offline_output.json") 
    print("'offline_output.json' 저장 완료")