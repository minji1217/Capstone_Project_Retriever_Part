# Recall@K, MRR 계산 (Retrieval 성능 점검 for 디버깅)
import numpy as np
import config

def calculate_metrics(predicted_ids, gt_ids, k_list):
    '''
    predicted_ids : RRF Score 통해 최종 top-k(100) 결과 리스트 
    gt_ids : 실제 저자가 인용한 정답 타겟 논문 ID 리스트 
    '''

    metrics = {}
    
