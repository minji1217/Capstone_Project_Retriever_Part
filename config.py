import os # 컴퓨터 파일이나 폴더 다룸 
# 1. 경로 설정 
'''
__file__ : 현재 실행 중인 이 파이썬 파일 자체 
abspath 통해 파일의 전체 주소를 절대 경로로 바꿔줌
os.path.dirname :  전체 주소에서 파일 이름은 빼고, 그 파일이 들어있는 폴더 주소만 가져옴
'''

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 기본 폴더 작성 
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "", "") # 실제 FAISS 인덱스 저장 파일 경로
ID_MAPPING_PATH = os.path.join(BASE_DIR, "", "") # 인덱스-논문 매핑 저장 파일 경로 

# 2. 모델/정규식
MODEL_NAME = "allenai/specter2_base"
ADAPTER_NAME = "allenai/specter2_proximity"
CITE_TAG_PATTERN = r"\[CITE:(.*?)\]"

# 3. Retrieval & Fusion 하이퍼파라미터 설정
WINDOW_SIZE = 100           # Context Query 생성시 placeholder 기준 자를 토큰 수 
SIMILARITY_THRESHOLD = 0.6  # FAISS 코사인 유사도 최소 임계값
TOP_K_RETRIEVAL = 75        # 1차 FAISS 검색에서 Paper/Context Query에 대해 관련 논문 각각 뽑을 개수 
TOP_K_FINAL = 100           # 75+75 -> fusion하여 최종 남길 후보 개수 
RRF_K = 60                  # RRF 스무딩 상수 
BATCH_SIZE = 16             # 배치 크기 
MAX_SEQ_LENGTH = 512        # SPECTER2 최대 입력 크기 
PAPER_SIM_WEIGHT = 0.4      # 중복 논문일 경우 가중합 비율 (paper_query) 
CONTEXT_SIM_WEIGHT = 0.6    # 중복 논문일 경우 가중합 비율 (context_query)
