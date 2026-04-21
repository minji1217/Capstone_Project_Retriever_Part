import json
import pickle # 파이썬 객체(리스트, 딕셔너리 등)를 있는 그대로 파일로 저장 or 다시 불러올 수 있게 해주는 lib


# pickle 파일은 이진데이터이므로 read binary 모드로 
# 이진파일을 원래의 파이썬 딕셔너리 형태로 변환 
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# 파이썬 객체를 Pickle 파일(이진 데이터)로 저장 
def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

    
# data : 저장하고 싶은 실제 내용(리스트, 딕셔너리 등)
# file_path : 어디에 어떤 이름으로 저장할지 알려주는 주소 (예: C:/.../a.json)
def save_json(data, file_path):
    with open(file_path, "w", encoding = "utf-8") as f: # utf-8 : 한글깨짐 방지
        # data를 f에 적을 때, 4칸씩 띄우고, 한글 깨지지 않게 저장 
        json.dump(data, f, indent = 4) # ensure_ascii = False는 한글 깨지지 않게 저장 

# JSON 파일을 파이썬 딕셔너리/리스트로 불러오기     
def load_json(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        return json.load(f)

