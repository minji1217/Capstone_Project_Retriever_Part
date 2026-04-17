import json
import pickle # 파이썬 객체(리스트, 딕셔너리 등)를 있는 그대로 파일로 저장 or 다시 불러올 수 있게 해주는 lib


# pickle 파일은 이진데이터이므로 read binary 모드로 
# 이진파일을 원래의 파이썬 딕셔너리 형태로 변환 
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
# data : 저장하고 싶은 실제 내용(리스트, 딕셔너리 등)
# file_path : 어디에 어떤 이름으로 저장할지 알려주는 주소 (예: C:/.../a.json)
def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f: # utf-8 : 한글깨짐 방지
        # data를 f에 적을 때, 4칸씩 띄우고, 한글 깨지지 않게 저장 
        json.dump(data, f, indent = 4) # ensure_ascii = False는 한글 깨지지 않게 저장 
        

def calculate_recall_at_k(retrieved_list, target_ids, k):
    '''
    retrieved_list : RRF로 정렬된 최종 후보 딕셔너리 리스트 Top-k(100) (예: [{"paper_id": "A", "score": 0.9, ...}, ...])
    target_ids: 실제 정답 논문 ID 리스트 ["A", "B"]
    '''
    # 0. 분모 0인 경우 에러 방지 
    if len(target_ids) == 0: return 0.0 

    # 1. top-k 후보 논문 ID 리스트 
    top_k_retrieved = [item["paper_id"] for item in retrieved_list[:k]]

    # 2. 분자 : Top-K 리스트와 정답지 간의 교집합 개수 세기 (for Recall)
    # -> 방법 : set 연산으로 교집합 구한 뒤 개수 세기 
    hit_count = len(set(top_k_retrieved).intersection(set(target_ids)))

    # 3. Recall 값 반환 (예: 정답 3개일 때 2개 맞히면 0.666...)

    return hit_count / len(target_ids)